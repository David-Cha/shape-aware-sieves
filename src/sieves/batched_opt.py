import copy
from gpytoolbox.copyleft import mesh_boolean
import kaolin as kal
from kaolin.rep import SurfaceMesh
import torch

from sieves import hole_filling as fill
from sieves import mesh_utils
from sieves import orientation as orie
from sieves import particle_swarm_opt as pso


def get_rasters(vtxs: torch.Tensor,
                faces: torch.Tensor,
                image_shape: tuple[int, int] = (256, 256),
                fill_holes: bool = False):
  '''
  Get Kaolin rasters from a batch of vertices corresponding to a fixed topology.

  Input:
  - vtxs: torch.Tensor of shape (..., V, 3)
      Batch of coordinates of a list of vertices.

  - faces: torch.Tensor of shape (F, 3)
      List of triplets of vertex indices forming triangle faces.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the rasters.

  - fill_holes: bool, optional (default False)
      Option to fill all holes in rasters.

  Output:
  - rasters: torch.Tensor of shape (..., x, y)
      Batch of rasters from rasterizing the batch of meshes specified by
      `vtxs` and `faces` where (x, y) equals `image_shape`.
  '''

  assert(len(vtxs.shape) >= 3 and vtxs.shape[-1] == 3)
  assert(len(faces.shape) == 2 and faces.shape[-1] == 3)

  camera_rot = torch.eye(3).unsqueeze(0).cuda()
  camera_trans = torch.zeros(3).unsqueeze(0).cuda()

  # Apply the transformation from camera_rot and camera_trans.
  vertices_camera = kal.render.camera.rotate_translate_points(vtxs, camera_rot, camera_trans)

  # Project the vertices on the camera image plan.
  face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(torch.flatten(vertices_camera, 0, -3), faces)
  face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)

  vtxs_flattened = torch.flatten(vtxs, 0, -3)
  vertices_image = vtxs_flattened[:,:,:2]
  face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)

  batch_size = vtxs_flattened.shape[0]
  num_faces = faces.shape[0]
  face_attributes = torch.ones(batch_size, num_faces, 3, 1).cuda()

  rendered_features, _, _ = kal.render.mesh.dibr_rasterization(image_shape[0],
                                                               image_shape[1],
                                                               face_vertices_camera[:, :, :, -1],
                                                               face_vertices_image,
                                                               face_attributes,
                                                               face_normals[:, :, -1])

  vtxs_batch_shape = vtxs.shape[:-2]
  rasters = torch.unflatten(rendered_features, 0, vtxs_batch_shape).flatten(-2)
  rasters = torch.clamp(rasters, 0., 1.)
  rasters[rasters > 0.] = 1.
  rasters = rasters.type(torch.IntTensor).cuda()

  if fill_holes:
    rasters = fill.get_filled_rasters(rasters)

  return rasters


def get_soft_masks(vtxs: torch.Tensor,
                   faces: torch.Tensor,
                   image_shape: tuple[int, int] = (256, 256),
                   fill_holes: bool = False):
  '''
  Get Kaolin soft mask rasters from a batch of vertices corresponding to a fixed topology.

  Input:
  - vtxs: torch.Tensor of shape (..., V, 3)
      Batch of coordinates of a list of vertices.

  - faces: torch.Tensor of shape (F, 3)
      List of triplets of vertex indices forming triangle faces.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  - fill_holes: bool, optional (default False)
      Option to fill all holes in soft masks.

  Output:
  - soft_mask: torch.Tensor of shape (..., x, y)
      Batch of soft masks from rasterizing the batch of meshes specified by
      `vtxs` and `faces` where (x, y) equals `image_shape`.
  '''

  assert(len(vtxs.shape) >= 3 and vtxs.shape[-1] == 3)
  assert(len(faces.shape) == 2 and faces.shape[-1] == 3)
  
  camera_rot = torch.eye(3).unsqueeze(0).cuda()
  camera_trans = torch.zeros(3).unsqueeze(0).cuda()
  
  # Apply the transformation from camera_rot and camera_trans.
  vertices_camera = kal.render.camera.rotate_translate_points(vtxs, camera_rot, camera_trans)
  
  # Project the vertices on the camera image plan.
  face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(torch.flatten(vertices_camera, 0, -3), faces)
  face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
  
  vtxs_flattened = torch.flatten(vtxs, 0, -3)
  vertices_image = vtxs_flattened[:,:,:2]
  face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)
  
  batch_size = vtxs_flattened.shape[0]
  num_faces = faces.shape[0]
  face_attributes = torch.ones(batch_size, num_faces, 3, 1).cuda()
  
  _, soft_mask, _ = kal.render.mesh.dibr_rasterization(image_shape[0],
                                                       image_shape[1],
                                                       face_vertices_camera[:, :, :, -1],
                                                       face_vertices_image,
                                                       face_attributes,
                                                       face_normals[:, :, -1])

  vtxs_batch_shape = vtxs.shape[:-2]
  soft_mask = torch.unflatten(soft_mask, 0, vtxs_batch_shape)

  if fill_holes:
    soft_mask = fill.get_filled_soft_masks(soft_mask)

  return soft_mask


def get_non_overlapping_areas(soft_masks_A: torch.Tensor,
                              soft_masks_B: torch.Tensor,
                              use_proportion: bool = True):
  '''
  For each A and B combination, get area of soft mask of B that
  is not in the intersection of the soft masks of A and B.

  Input:
  - soft_masks_A: torch.Tensor of shape (a, x, y)
      Soft masks for `a` orientations of mesh A.

  - soft_masks_B: torch.Tensor of shape (a, b, x, y)
      Soft masks for `a * b` orientations of mesh B.

  - use_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  Output:
  - non_overlapping_areas: torch.Tensor of shape (a, b)
      Areas of soft masks of B not in the intersection of the soft
      masks of A and B for each possible combination of soft masks
      from `soft_masks_A` and `soft_masks_B`.
  '''

  assert(soft_masks_A.shape[0] == soft_masks_B.shape[0])

  # Get projection areas of each orientation of B.
  soft_masks_B_areas = torch.sum(soft_masks_B, dim=(-2, -1))

  # Get projection intersection areas of each orientation of B with each orientation of A.
  soft_masks_intersections = soft_masks_A.unsqueeze(1) * soft_masks_B
  soft_masks_intersections_areas = torch.sum(soft_masks_intersections, dim=(-2, -1))

  # Get non-overlapping areas of B's projection for each orientation of B with each orientation of A.
  non_overlapping_areas = soft_masks_B_areas - soft_masks_intersections_areas

  if use_proportion:
    assert(torch.all(soft_masks_B_areas > 0.))
    non_overlapping_areas = torch.div(non_overlapping_areas, soft_masks_B_areas)

  return non_overlapping_areas


def get_out_of_bounds_vtxs_losses(vtxs: torch.Tensor,
                                  weight: float = 1000.):
  '''
  Compute out of bounds loss for each set of vertices in a batch of vertices.

  Input:
  - vtxs: torch.Tensor of shape (..., V, 3)
      Batch of coordinates of a list of vertices.

  - weight: float, optional (default 1000.0)
      Value to multiply resulting loss by.

  Output:
  - weighted_total_losses: torch.Tensor of shape (...,)
      Batch of out-of-bounds vertices losses.
  '''
  
  # Compute out of bounds loss for x-coordinates per mesh.
  x_coords = vtxs[...,0]
  x_losses = torch.sum(torch.max(torch.abs(x_coords), torch.ones(x_coords.shape).cuda()) - 1., dim=-1)
  
  # Compute out of bounds loss for y-coordinates per mesh.
  y_coords = vtxs[...,1]
  y_losses = torch.sum(torch.max(torch.abs(y_coords), torch.ones(y_coords.shape).cuda()) - 1., dim=-1)
  
  total_losses = x_losses + y_losses
  weighted_total_losses = weight * total_losses

  return weighted_total_losses


def get_loss_per_B_posn_per_A_posn(soft_mask_per_A_posn: torch.Tensor,
                                   vtxs_B: torch.Tensor,
                                   faces_B: torch.Tensor,
                                   rot_vec_per_B_posn_per_A_posn: torch.Tensor,
                                   trans_vec_per_B_posn_per_A_posn: torch.Tensor,
                                   use_area_proportion: bool = True,
                                   use_hard_masks: bool = False,
                                   out_of_bounds_vtxs_weight: float = 1000.,
                                   image_shape: tuple[int, int] = (256, 256)):
  '''
  Given a batch of soft masks of mesh A corresponding to multiple different orientations of A
  and a batch of orientations for mesh B, compute the min loss for each A across all B.

  Input:
  - soft_mask_per_A_posn: torch.Tensor of shape (p_a, x, y)
      Batch of soft masks of mesh A, each corresponding to a different orientation.

  - vtxs_B: torch.Tensor of shape (V, 3)
      Vertex coordinates for mesh B.

  - faces_B: torch.Tensor of shape (F, 3)
      Vertex indices for faces of mesh B.

  - rot_vec_per_B_posn_per_A_posn: torch.Tensor of shape (p_a, p_b, 6)
      Rotation vectors for each orientation of mesh B for each orientation of mesh A.

  - trans_vec_per_B_posn_per_A_posn: torch.Tensor of shape (p_a, p_b, 2)
      Translation vectors for each orientation of mesh B for each orientation of mesh A.

  - use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - use_hard_masks: bool, optional (default False)
      Option to use binary rasters of mesh B for all computations.
      If `True`, `soft_mask_per_A_posn` should also be binary rasters of mesh A.

  - out_of_bounds_vtxs_weight: float, optional (default 1000.0)
      Value to multiply loss calculated in `get_out_of_bounds_vtxs_losses()`.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - total_loss_per_B_posn_per_A_posn: torch.Tensor of shape (p_a, p_b)
      Total loss for each orientation of mesh B for each orientation of mesh A.
  '''

  vtxs_B_transformed_per_B_posn_per_A_posn = orie.get_transformed_vtxs(vtxs_B, rot_vec_per_B_posn_per_A_posn, trans_vec_per_B_posn_per_A_posn)
  soft_mask_per_B_posn_per_A_posn = get_rasters(vtxs_B_transformed_per_B_posn_per_A_posn, faces_B, image_shape, False) \
                                    if use_hard_masks else get_soft_masks(vtxs_B_transformed_per_B_posn_per_A_posn, faces_B, image_shape, False)
  
  overlap_loss_per_B_posn_per_A_posn = get_non_overlapping_areas(soft_mask_per_A_posn, soft_mask_per_B_posn_per_A_posn, use_area_proportion)
  out_of_bounds_loss_per_B_posn_per_A_posn = get_out_of_bounds_vtxs_losses(vtxs_B_transformed_per_B_posn_per_A_posn, out_of_bounds_vtxs_weight)
  total_loss_per_B_posn_per_A_posn = overlap_loss_per_B_posn_per_A_posn + out_of_bounds_loss_per_B_posn_per_A_posn

  return total_loss_per_B_posn_per_A_posn


def get_loss_per_B_posn_per_A_posn_per_B_mesh(soft_mask_per_A_posn: torch.Tensor,
                                              meshes_B: list[SurfaceMesh],
                                              rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                              trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                              use_area_proportion: bool = True,
                                              use_hard_masks: bool = False,
                                              out_of_bounds_vtxs_weight: float = 1000.,
                                              image_shape: tuple[int, int] = (256, 256)):
  '''
  Given a batch of soft masks of mesh A corresponding to multiple different orientations of A
  and a batch of orientations for mesh B, compute the loss for each A and B pair for each mesh B
  and return the min for each orientation of A.

  Input:
  - soft_masks_A: torch.Tensor of shape (p_a, x, y)
      Batch of soft masks of mesh A, each corresponding to a different orientation.

  - meshes_B: list of kaolin.rep.SurfaceMesh
      List of meshes to block.

  - rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n_b, p_a, p_b, 6)
      Rotation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n_b, p_a, p_b, 2)
      Translation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - use_hard_masks: bool, optional (default False)
      Option to use binary rasters of mesh B for all computations.
      If `True`, `soft_mask_per_A_posn` should also be binary rasters of mesh A.

  - out_of_bounds_vtxs_weight: float, optional (default 1000.0)
      Value to multiply loss calculated in `get_out_of_bounds_vtxs_losses()`.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - loss_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n_b, p_a, p_b)
      Total loss for each orientation B for each orientation of mesh A for each mesh B.
  '''

  assert(rot_vec_per_B_posn_per_A_posn_per_B_mesh.shape[:2] == trans_vec_per_B_posn_per_A_posn_per_B_mesh.shape[:2])
  assert(len(meshes_B) == rot_vec_per_B_posn_per_A_posn_per_B_mesh.shape[0])
  assert(soft_mask_per_A_posn.shape[0] == rot_vec_per_B_posn_per_A_posn_per_B_mesh.shape[1])

  loss_per_B_posn_per_A_posn_per_B_mesh = torch.zeros(rot_vec_per_B_posn_per_A_posn_per_B_mesh.shape[0],
                                                      rot_vec_per_B_posn_per_A_posn_per_B_mesh.shape[1],
                                                      rot_vec_per_B_posn_per_A_posn_per_B_mesh.shape[2]).cuda()

  for mesh_B_idx, mesh_B in enumerate(meshes_B):
    rot_vec_per_B_posn_per_A_posn = rot_vec_per_B_posn_per_A_posn_per_B_mesh[mesh_B_idx]
    trans_vec_per_B_posn_per_A_posn = trans_vec_per_B_posn_per_A_posn_per_B_mesh[mesh_B_idx]

    loss_per_B_posn_per_A_posn_per_B_mesh[mesh_B_idx] = get_loss_per_B_posn_per_A_posn(soft_mask_per_A_posn,
                                                                                       mesh_B.vertices,
                                                                                       mesh_B.faces,
                                                                                       rot_vec_per_B_posn_per_A_posn,
                                                                                       trans_vec_per_B_posn_per_A_posn,
                                                                                       use_area_proportion,
                                                                                       use_hard_masks,
                                                                                       out_of_bounds_vtxs_weight,
                                                                                       image_shape)

  return loss_per_B_posn_per_A_posn_per_B_mesh


def optimize_B_posn_per_A_posn_per_B_mesh(rot_vec_per_A_posn: torch.Tensor,
                                          mesh_A: SurfaceMesh,
                                          meshes_B: list[SurfaceMesh],
                                          rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                          trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                          mesh_A_fill_soft_mask_holes: bool = True,
                                          mesh_B_opt_use_area_proportion: bool = True,
                                          mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                                          mesh_B_opt_iters: int = 100,
                                          mesh_B_opt_learning_rate: float = 0.05,
                                          losses: torch.Tensor = None,
                                          image_shape: tuple[int, int] = (256, 256)):
  '''
  Given initial orientations for the B meshes, optimize them against the current positions of mesh A.

  Input:
  - rot_vec_per_A_posn: torch.Tensor of shape (a, 6)
      List of rotations for mesh A.

  - mesh_A: kaolin.rep.SurfaceMesh
      The mesh to optimize the orientation of so that its projection blocks B meshes.

  - meshes_B: list[kaolin.rep.SurfaceMesh]
      List of meshes to block.

  - rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 6)
      Rotation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 2)
      Translation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes in soft masks of mesh A.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - loss_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b)
      Optimized losses per orientation.

  - opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 6)
      Optimized rotation vectors per orientation.

  - opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 2)
      Optimized translation vectors per orientation.
  '''

  if losses is None:
    losses = torch.zeros(mesh_B_opt_iters + 1).cuda()

  vtxs_A_transformed_per_A_posn = orie.get_rotated_vtxs(mesh_A.vertices, rot_vec_per_A_posn.detach().clone())
  soft_mask_per_A_posn = get_soft_masks(vtxs_A_transformed_per_A_posn, mesh_A.faces, image_shape, mesh_A_fill_soft_mask_holes)

  opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn_per_A_posn_per_B_mesh.detach().clone().requires_grad_(True)
  opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh = trans_vec_per_B_posn_per_A_posn_per_B_mesh.detach().clone().requires_grad_(True)

  params = [opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh, opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh]
  optim = torch.optim.Adam(params=params, lr=mesh_B_opt_learning_rate)

  for iter in range(mesh_B_opt_iters):
    loss_per_B_posn_per_A_posn_per_B_mesh = get_loss_per_B_posn_per_A_posn_per_B_mesh(soft_mask_per_A_posn,
                                                                                      meshes_B,
                                                                                      opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                      opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                      mesh_B_opt_use_area_proportion,
                                                                                      False,
                                                                                      mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                      image_shape)

    loss = torch.sum(loss_per_B_posn_per_A_posn_per_B_mesh)

    losses[iter] = loss.item()

    optim.zero_grad()
    loss.backward()
    optim.step()

  loss_per_B_posn_per_A_posn_per_B_mesh = get_loss_per_B_posn_per_A_posn_per_B_mesh(soft_mask_per_A_posn,
                                                                                    meshes_B,
                                                                                    opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                    opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                    mesh_B_opt_use_area_proportion,
                                                                                    False,
                                                                                    mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                    image_shape)

  losses[mesh_B_opt_iters] = torch.sum(loss_per_B_posn_per_A_posn_per_B_mesh).item()

  return loss_per_B_posn_per_A_posn_per_B_mesh.detach().clone(), \
         opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh.detach().clone(), \
         opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh.detach().clone()


def optimize_posn_of_A_pso_objective(rot_vec_per_A_posn: torch.Tensor,
                                     mesh_A: SurfaceMesh,
                                     meshes_B: list[SurfaceMesh],
                                     rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                     trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                     mesh_A_fill_soft_mask_holes: bool = True,
                                     mesh_B_opt_use_area_proportion: bool = True,
                                     mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                                     mesh_B_opt_iters: int = 100,
                                     mesh_B_opt_learning_rate: float = 0.05,
                                     image_shape: tuple[int, int] = (256, 256)):
  '''
  Objective function to be passed into particle swarm optimizer for optimizing orientation
  of mesh A to block B meshes.

  Input:
  - rot_vec_per_A_posn: torch.Tensor of shape (a, 6)
      List of rotations for mesh A.

  - mesh_A: kaolin.rep.SurfaceMesh
      The mesh to optimize the orientation of so that its projection blocks B meshes.

  - meshes_B: list[kaolin.rep.SurfaceMesh]
      List of meshes to block.

  - rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 6)
      Rotation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 2)
      Translation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes in soft masks of mesh A.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - loss_per_A_posn: torch.Tensor of shape (a,)
      Total loss per orientation of mesh A.
  '''

  _, opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh, opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh = optimize_B_posn_per_A_posn_per_B_mesh(rot_vec_per_A_posn=rot_vec_per_A_posn,
                                                                                                                                          mesh_A=mesh_A,
                                                                                                                                          meshes_B=meshes_B,
                                                                                                                                          rot_vec_per_B_posn_per_A_posn_per_B_mesh=rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                                                                          trans_vec_per_B_posn_per_A_posn_per_B_mesh=trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                                                                          mesh_A_fill_soft_mask_holes=mesh_A_fill_soft_mask_holes,
                                                                                                                                          mesh_B_opt_use_area_proportion=mesh_B_opt_use_area_proportion,
                                                                                                                                          mesh_B_opt_vtx_out_of_bounds_loss_weight=mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                                                                          mesh_B_opt_iters=mesh_B_opt_iters,
                                                                                                                                          mesh_B_opt_learning_rate=mesh_B_opt_learning_rate,
                                                                                                                                          image_shape=image_shape)

  # Compute final loss per A after last optimization step using hard masks.
  vtxs_A_transformed_per_A_posn = orie.get_rotated_vtxs(mesh_A.vertices, rot_vec_per_A_posn.detach().clone())
  hard_mask_per_A_posn = get_rasters(vtxs_A_transformed_per_A_posn, mesh_A.faces, image_shape, mesh_A_fill_soft_mask_holes)
  loss_per_B_posn_per_A_posn_per_B_mesh = get_loss_per_B_posn_per_A_posn_per_B_mesh(hard_mask_per_A_posn,
                                                                                    meshes_B,
                                                                                    opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                    opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                    mesh_B_opt_use_area_proportion,
                                                                                    True,
                                                                                    mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                    image_shape)

  loss_per_A_posn_per_B_mesh = loss_per_B_posn_per_A_posn_per_B_mesh.min(dim=-1).values
  loss_per_A_posn = loss_per_A_posn_per_B_mesh.T.min(dim=-1).values

  return loss_per_A_posn.detach().clone()


def optimize_posn_of_A(mesh_A: SurfaceMesh,
                       meshes_B: list[SurfaceMesh],
                       mesh_A_num_posns: int = 10,
                       mesh_B_num_posns: int = 10,
                       mesh_A_fill_soft_mask_holes: bool = True,
                       mesh_B_opt_posn_trans_bounds: tuple[float, float] = (-0.1, 0.1),
                       mesh_B_opt_use_area_proportion: bool = True,
                       mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                       mesh_B_opt_iters: int = 100,
                       mesh_B_opt_learning_rate: float = 0.05,
                       pso_c1: float = 0.25,
                       pso_c2: float = 0.25,
                       pso_w: float = 0.5,
                       pso_iters: int = 10,
                       pso_verbose: bool = False,
                       image_shape: tuple[int, int] = (256, 256)):
  '''
  Find optimized orientation of mesh A against a set of B meshes using particle swarm optimization (PSO).

  Input:
  - mesh_A: kaolin.rep.SurfaceMesh
      The mesh to optimize the orientation of so that its projection blocks B meshes.

  - meshes_B: list[kaolin.rep.SurfaceMesh]
      List of meshes to block.

  - mesh_A_num_posns: int, optional (default 10)
      Number of particles, that is, orientations of mesh A, to use for PSO.

  - mesh_B_num_posns: int, optional (default 10)
      Number of initial orientations to use when optimizing a B mesh against mesh A.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes in soft masks of mesh A.

  - mesh_B_opt_posn_trans_bounds: tuple[float, float], optional (default (-0.1, 0.1))
      Bounds for randomly generated translation vectors for the initial orientations of mesh B.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - pso_c1: float, optional (default 0.25)
      PSO cognitive coefficient.

  - pso_c2: float, optional (default 0.25)
      PSO social coefficient.

  - pso_w: float, optional (default 0.5)
      PSO inertia weight.

  - pso_iters: int, optional (default 10)
      Number of PSO iterations to run.

  - pso_verbose: bool, optional (default False)
      Option to print logs during PSO.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - best_val: float
      Best loss achieved during PSO.

  - best_posn: torch.Tensor of shape (6,)
      Orientation of mesh A corresponding to the best loss achieved during PSO.
  '''

  init_A_posns = orie.generate_rotation_vectors(mesh_A_num_posns).cuda()

  rot_vec_per_B_posn = orie.generate_rotation_vectors(mesh_B_num_posns)
  trans_vec_per_B_posn = orie.generate_uniform_random_vectors(mesh_B_num_posns, 2, mesh_B_opt_posn_trans_bounds[0], mesh_B_opt_posn_trans_bounds[1])

  num_B_meshes = len(meshes_B)
  repeat_shape = [num_B_meshes, mesh_A_num_posns]
  rot_vec_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(rot_vec_per_B_posn.shape))])
  trans_vec_per_B_posn_per_A_posn_per_B_mesh = trans_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(trans_vec_per_B_posn.shape))])

  kwargs = {
    'mesh_A': mesh_A,
    'meshes_B': meshes_B,
    'rot_vec_per_B_posn_per_A_posn_per_B_mesh': rot_vec_per_B_posn_per_A_posn_per_B_mesh,
    'trans_vec_per_B_posn_per_A_posn_per_B_mesh': trans_vec_per_B_posn_per_A_posn_per_B_mesh,
    'mesh_A_fill_soft_mask_holes': mesh_A_fill_soft_mask_holes,
    'mesh_B_opt_use_area_proportion': mesh_B_opt_use_area_proportion,
    'mesh_B_opt_vtx_out_of_bounds_loss_weight': mesh_B_opt_vtx_out_of_bounds_loss_weight,
    'mesh_B_opt_iters': mesh_B_opt_iters,
    'mesh_B_opt_learning_rate': mesh_B_opt_learning_rate,
    'image_shape': image_shape
  }

  pso_optim = pso.ParticleSwarmOptimizer(num_particles=mesh_A_num_posns,
                                        dim=init_A_posns.shape[1],
                                        c1=pso_c1,
                                        c2=pso_c2,
                                        w=pso_w,
                                        init_pos=init_A_posns,
                                        obj_func=optimize_posn_of_A_pso_objective,
                                        obj_func_kwargs=kwargs)

  best_val, best_posn = pso_optim.optimize(num_iters=pso_iters, verbose=pso_verbose)

  return best_val, best_posn


def optimize_winning_posn_of_mesh_A_to_overlap_base_mesh_soft_mask(base_mesh_soft_mask: torch.Tensor,
                                                                   mesh_A: SurfaceMesh,
                                                                   mesh_A_num_posns: int = 10,
                                                                   mesh_A_posn_rot_bounds: tuple[float, float] = (0., 2. * torch.pi),
                                                                   mesh_A_posn_trans_bounds: tuple[float, float] = (-0.1, 0.1),
                                                                   mesh_A_fill_soft_mask_holes: bool = True,
                                                                   mesh_A_opt_use_area_proportion: bool = True,
                                                                   mesh_A_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                                                                   mesh_A_opt_iters: int = 100,
                                                                   mesh_A_opt_learning_rate: float = 0.05,
                                                                   image_shape: tuple[int, int] = (256, 256)):
  '''
  Optimize orientation of mesh A using only 2D transformations (rotations around Z-axis plus
  translations along X-axis and Y-axis) to maximize overlap of its projection soft mask with
  that of a base mesh.

  Input:
  - base_mesh_soft_mask: torch.Tensor of shape (1, image_shape[0], image_shape[1])
      Soft mask to maximize overlap with.

  - mesh_A: kaolin.rep.SurfaceMesh
      Mesh to optimize orientation of with 2D transformations.

  - mesh_A_num_posns: int, optional (default 10)
      Number of initial starting orientations of mesh A.

  - mesh_A_posn_rot_bounds: tuple[float, float], optional (default (0.0, 2.0 * torch.pi))
      Bounds for rotation angles of mesh A orientations.

  - mesh_A_posn_trans_bounds: tuple[float, float], optional (default (-0.1, 0.1))
      Bounds for translation vectors of mesh A orientations.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes of soft mask of mesh A.

  - mesh_A_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of mesh A or just the area as is.

  - mesh_A_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_A_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of mesh A.

  - mesh_A_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of mesh A

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - opt_angle_per_A_posn: torch.Tensor of shape (mesh_A_num_posns,)
      Angles of all optimized orientations.

  - opt_trans_vec_per_A_posn: torch.Tensor of shape (mesh_A_num_posns,)
      Translation vectors of all optimized orientations.
  '''

  mesh_A_vtxs_xy = mesh_A.vertices[:,:2]

  opt_angle_per_A_posn = orie.generate_uniform_random_vectors(mesh_A_num_posns,
                                                              1,
                                                              mesh_A_posn_rot_bounds[0],
                                                              mesh_A_posn_rot_bounds[1]).detach().clone().flatten().requires_grad_(True)
  opt_trans_vec_per_A_posn = orie.generate_uniform_random_vectors(mesh_A_num_posns,
                                                                  2,
                                                                  mesh_A_posn_trans_bounds[0],
                                                                  mesh_A_posn_trans_bounds[1]).detach().clone().requires_grad_(True)
  params = [opt_angle_per_A_posn, opt_trans_vec_per_A_posn]
  optim = torch.optim.Adam(params=params, lr=mesh_A_opt_learning_rate)

  for _ in range(mesh_A_opt_iters):
    rot_mat_per_A_posn = orie.get_z_axis_rot_mats(opt_angle_per_A_posn)
    mesh_A_vtxs_xy_transformed_per_A_posn = torch.transpose(torch.matmul(rot_mat_per_A_posn, torch.transpose(mesh_A_vtxs_xy, -2, -1)), -2, -1) \
                                           + opt_trans_vec_per_A_posn.unsqueeze(-2)
    mesh_A_vtxs_transformed_per_A_posn = mesh_A.vertices.repeat(mesh_A_num_posns, 1, 1)
    mesh_A_vtxs_transformed_per_A_posn[:,:,:2] = mesh_A_vtxs_xy_transformed_per_A_posn
    soft_mask_per_A_posn = get_soft_masks(mesh_A_vtxs_transformed_per_A_posn, mesh_A.faces, image_shape, mesh_A_fill_soft_mask_holes).unsqueeze(0)

    overlap_loss_per_A_posn = get_non_overlapping_areas(base_mesh_soft_mask, soft_mask_per_A_posn, mesh_A_opt_use_area_proportion)
    out_of_bounds_loss_per_A_posn = get_out_of_bounds_vtxs_losses(mesh_A_vtxs_transformed_per_A_posn, mesh_A_opt_vtx_out_of_bounds_loss_weight)
    loss_per_A_posn = overlap_loss_per_A_posn + out_of_bounds_loss_per_A_posn

    loss = torch.sum(loss_per_A_posn)

    optim.zero_grad()
    loss.backward()
    optim.step()
  
  return opt_angle_per_A_posn.detach().clone(), opt_trans_vec_per_A_posn.detach().clone()


def optimize_winning_posn_of_mesh_A_with_base_mesh_pso_objective(mesh_A_posns: torch.Tensor,
                                                                 base_mesh: SurfaceMesh,
                                                                 base_mesh_hard_mask_area: float,
                                                                 mesh_A: SurfaceMesh,
                                                                 meshes_B: list[SurfaceMesh],
                                                                 rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                                                 trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                                                 mesh_A_fill_soft_mask_holes: bool = True,
                                                                 mesh_B_opt_use_area_proportion: bool = True,
                                                                 mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                                                                 mesh_B_opt_iters: int = 100,
                                                                 mesh_B_opt_learning_rate: float = 0.05,
                                                                 image_shape: tuple[int, int] = (256, 256)):
  '''
  Objective function to be passed into particle swarm optimizer for optimizing the
  orientation of mesh A with a base mesh so that the projection of the union of
  those meshes block a set of B meshes.

  Input:
  - mesh_A_posns: torch.Tensor of shape (a, 3)
      Current orientations of mesh A.

  - base_mesh: kaolin.rep.SurfaceMesh
      Mesh to form union with mesh A.

  - base_mesh_hard_mask_area: float
      Area of projection of base mesh.

  - mesh_A: kaolin.rep.SurfaceMesh
      Mesh to optimize orientation of with 2D transformations.

  - meshes_B: list[kaolin.rep.SurfaceMesh]
      Meshes to block.

  - rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 6)
      Rotation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor of shape (n, a, b, 2)
      Translation vectors for each orientation of mesh B for each orientation of mesh A for each mesh B.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes of soft mask of mesh A.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - total_loss_per_A_posn: torch.Tensor of shape (a,)
      Loss per orientation of mesh A.
  '''

  base_mesh_V = base_mesh.vertices.detach().clone().cpu().numpy()
  base_mesh_F = base_mesh.faces.detach().clone().cpu().numpy()

  mesh_A_num_posns = mesh_A_posns.shape[0]
  mesh_A_angle_per_A_posn = mesh_A_posns[:,0]
  mesh_A_trans_vec_per_A_posn = mesh_A_posns[:,1:]
  mesh_A_vtxs_xy = mesh_A.vertices[:,:2].detach().clone()
  mesh_A_vtxs_xy_transformed_per_A_posn = orie.get_transformed_vtxs_2D(mesh_A_vtxs_xy, mesh_A_angle_per_A_posn, mesh_A_trans_vec_per_A_posn)
  mesh_A_vtxs_transformed_per_A_posn = mesh_A.vertices.detach().clone().repeat(mesh_A_num_posns, 1, 1)
  mesh_A_vtxs_transformed_per_A_posn[:,:,:2] = mesh_A_vtxs_xy_transformed_per_A_posn

  union_mesh_soft_mask_per_A_posn = torch.zeros(mesh_A_num_posns, image_shape[0], image_shape[1]).cuda()
  union_mesh_hard_mask_per_A_posn = torch.zeros(mesh_A_num_posns, image_shape[0], image_shape[1]).cuda()

  for posn_idx in range(mesh_A_num_posns):
    mesh_A_transformed_V = mesh_A_vtxs_transformed_per_A_posn[posn_idx].detach().clone().cpu().numpy()
    mesh_A_transformed_F = mesh_A.faces.detach().clone().cpu().numpy()

    union_mesh_V, union_mesh_F = mesh_boolean(base_mesh_V,
                                              base_mesh_F,
                                              mesh_A_transformed_V,
                                              mesh_A_transformed_F,
                                              boolean_type='union')
    union_mesh_V = torch.FloatTensor(union_mesh_V).cuda()
    union_mesh_F = torch.LongTensor(union_mesh_F).cuda()

    union_mesh_soft_mask = get_soft_masks(union_mesh_V.unsqueeze(0),
                                          union_mesh_F,
                                          image_shape,
                                          mesh_A_fill_soft_mask_holes)[0]
    union_mesh_soft_mask_per_A_posn[posn_idx] = union_mesh_soft_mask

    union_mesh_hard_mask = get_rasters(union_mesh_V.unsqueeze(0),
                                       union_mesh_F,
                                       image_shape,
                                       mesh_A_fill_soft_mask_holes)[0]
    union_mesh_hard_mask_per_A_posn[posn_idx] = union_mesh_hard_mask

  opt_rot_vecs_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn_per_A_posn_per_B_mesh.detach().clone().requires_grad_(True)
  opt_trans_vecs_per_B_per_A_posn_per_B_mesh = trans_vec_per_B_posn_per_A_posn_per_B_mesh.detach().clone().requires_grad_(True)
  params = [opt_rot_vecs_per_B_posn_per_A_posn_per_B_mesh, opt_trans_vecs_per_B_per_A_posn_per_B_mesh]
  optim = torch.optim.Adam(params=params, lr=mesh_B_opt_learning_rate)

  for _ in range(mesh_B_opt_iters):
    blocking_loss_per_B_posn_per_A_posn_per_B_mesh = get_loss_per_B_posn_per_A_posn_per_B_mesh(union_mesh_soft_mask_per_A_posn,
                                                                                               meshes_B,
                                                                                               opt_rot_vecs_per_B_posn_per_A_posn_per_B_mesh,
                                                                                               opt_trans_vecs_per_B_per_A_posn_per_B_mesh,
                                                                                               mesh_B_opt_use_area_proportion,
                                                                                               False,
                                                                                               mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                               image_shape)

    loss = torch.sum(blocking_loss_per_B_posn_per_A_posn_per_B_mesh)

    optim.zero_grad()
    loss.backward()
    optim.step()

  # Compute final loss per A after last optimization step using hard masks.
  blocking_loss_per_B_posn_per_A_posn_per_B_mesh = get_loss_per_B_posn_per_A_posn_per_B_mesh(union_mesh_hard_mask_per_A_posn,
                                                                                             meshes_B,
                                                                                             opt_rot_vecs_per_B_posn_per_A_posn_per_B_mesh,
                                                                                             opt_trans_vecs_per_B_per_A_posn_per_B_mesh,
                                                                                             mesh_B_opt_use_area_proportion,
                                                                                             True,
                                                                                             mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                             image_shape).detach().clone()
  blocking_loss_per_A_posn_per_B_mesh = blocking_loss_per_B_posn_per_A_posn_per_B_mesh.min(dim=-1).values
  blocking_loss_per_A_posn = blocking_loss_per_A_posn_per_B_mesh.T.min(dim=-1).values
  blocking_loss_per_A_posn_activated = 0.5 * torch.tanh(200 * (blocking_loss_per_A_posn - 0.02)) + 0.5
  blocking_loss_per_A_posn_weight = 10.

  # Compute loss to minimize total area.
  union_mesh_hard_masks_area_per_A_posn = torch.sum(union_mesh_hard_mask_per_A_posn, dim=(-2, -1))
  area_loss_per_A_posn = union_mesh_hard_masks_area_per_A_posn / base_mesh_hard_mask_area - 1.
  area_loss_per_A_posn_weight = -5.

  total_loss_per_A_posn = blocking_loss_per_A_posn_weight * blocking_loss_per_A_posn_activated + area_loss_per_A_posn_weight * area_loss_per_A_posn

  return total_loss_per_A_posn


def optimize_winning_posn_of_mesh_A_with_base_mesh(base_mesh: SurfaceMesh,
                                                   mesh_A: SurfaceMesh,
                                                   meshes_B: list[SurfaceMesh],
                                                   base_mesh_fill_soft_mask_holes: bool = True,
                                                   mesh_A_num_posns: int = 10,
                                                   mesh_A_fill_soft_mask_holes: bool = True,
                                                   mesh_B_opt_num_posns: int = 10,
                                                   mesh_B_opt_posn_trans_bounds: tuple[float, float] = (-0.1, 0.1),
                                                   mesh_B_opt_use_area_proportion: bool = True,
                                                   mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                                                   mesh_B_opt_iters: int = 100,
                                                   mesh_B_opt_learning_rate: float = 0.05,
                                                   pso_c1: float = 0.05,
                                                   pso_c2: float = 0.05,
                                                   pso_w: float = 0.5,
                                                   pso_iters: int = 10,
                                                   pso_verbose: bool = False,
                                                   image_shape: tuple[int, int] = (256, 256)):
  '''
  Given a winning orientation of mesh A against a set of B meshes, optimize its orientation using
  only 2D transformations (rotation around Z-axis plus translations across X-axis and Y-axis) so
  that the projection of the union of mesh A and a base mesh blocks all B meshes while also
  minimizing total area.

  Input:
  - base_mesh: kaolin.rep.SurfaceMesh
      Mesh to form union with mesh A.

  - mesh_A: kaolin.rep.SurfaceMesh
      Mesh to optimize orientation of with 2D transformations.

  - meshes_B: list[kaolin.rep.SurfaceMesh]
      Meshes to block.

  - base_mesh_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes of soft mask of base mesh.

  - mesh_A_num_posns: int, optional (default 10)
      Number of random positions, or particles, of mesh A to use for PSO.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes of soft mask of mesh A.

  - mesh_B_opt_num_posns: int, optional (default 10)
      Number of randomly generated positions to use whenever optimizing orientation of a B mesh.

  - mesh_B_opt_posn_trans_bounds: tuple[float, float], optional (default (-0.1, 0.1))
      Bounds for translation vectors of the randomly generated positions for B meshes.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - pso_c1: float, optional (default 0.05)
      PSO cognitive coefficient.

  - pso_c2: float, optional (default 0.05)
      PSO social coefficient.

  - pso_w: float, optional (default 0.5)
      PSO inertia weight.

  - pso_iters: int, optional (default 10)
      Number of PSO iterations to run.

  - pso_verbose: bool, optional (default False)
      Option to print logs during PSO.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - bes_val: float
      Best objective value achieved.

  - best_posn_angle: float
      Angle of best orientation achieved.

  - best_posn_trans_vec: torch.Tensor of shape (2,)
      Translation vector of best orientation achieved.
  '''

  base_mesh_soft_mask = get_soft_masks(base_mesh.vertices.unsqueeze(0), base_mesh.faces, image_shape, base_mesh_fill_soft_mask_holes)
  mesh_A_init_angle_per_A_posn, mesh_A_init_trans_vec_per_A_posn = optimize_winning_posn_of_mesh_A_to_overlap_base_mesh_soft_mask(base_mesh_soft_mask,
                                                                                                                                  mesh_A,
                                                                                                                                  mesh_A_num_posns,
                                                                                                                                  image_shape=image_shape)
  mesh_A_init_posns = torch.cat([mesh_A_init_angle_per_A_posn.unsqueeze(1), mesh_A_init_trans_vec_per_A_posn], dim=-1)

  rot_vec_per_B_posn = orie.generate_rotation_vectors(mesh_B_opt_num_posns)
  trans_vec_per_B_posn = orie.generate_uniform_random_vectors(mesh_B_opt_num_posns,
                                                              2,
                                                              mesh_B_opt_posn_trans_bounds[0],
                                                              mesh_B_opt_posn_trans_bounds[1])
  num_B_meshes = len(meshes_B)
  repeat_shape = [num_B_meshes, mesh_A_init_posns.shape[0]]
  rot_vec_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(rot_vec_per_B_posn.shape))])
  trans_vec_per_B_posn_per_A_posn_per_B_mesh = trans_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(trans_vec_per_B_posn.shape))])

  base_mesh_hard_mask = get_rasters(base_mesh.vertices.unsqueeze(0), base_mesh.faces, image_shape, base_mesh_fill_soft_mask_holes)
  base_mesh_hard_mask_area = torch.sum(base_mesh_hard_mask[0], dim=(-2, -1))

  kwargs = {
    'base_mesh': base_mesh,
    'base_mesh_hard_mask_area': base_mesh_hard_mask_area,
    'mesh_A': mesh_A,
    'meshes_B': meshes_B,
    'rot_vec_per_B_posn_per_A_posn_per_B_mesh': rot_vec_per_B_posn_per_A_posn_per_B_mesh,
    'trans_vec_per_B_posn_per_A_posn_per_B_mesh': trans_vec_per_B_posn_per_A_posn_per_B_mesh,
    'mesh_A_fill_soft_mask_holes': mesh_A_fill_soft_mask_holes,
    'mesh_B_opt_use_area_proportion': mesh_B_opt_use_area_proportion,
    'mesh_B_opt_vtx_out_of_bounds_loss_weight': mesh_B_opt_vtx_out_of_bounds_loss_weight,
    'mesh_B_opt_iters': mesh_B_opt_iters,
    'mesh_B_opt_learning_rate': mesh_B_opt_learning_rate,
    'image_shape': image_shape
  }

  pso_optim = pso.ParticleSwarmOptimizer(num_particles=mesh_A_init_posns.shape[0],
                                        dim=mesh_A_init_posns.shape[1],
                                        c1=pso_c1,
                                        c2=pso_c2,
                                        w=pso_w,
                                        init_pos=mesh_A_init_posns,
                                        obj_func=optimize_winning_posn_of_mesh_A_with_base_mesh_pso_objective,
                                        obj_func_kwargs=kwargs)

  best_val, best_posn = pso_optim.optimize(num_iters=pso_iters, verbose=pso_verbose)
  best_posn_angle = best_posn[0]
  best_posn_trans_vec = best_posn[1:]

  return best_val, best_posn_angle, best_posn_trans_vec


def optimize_posn_of_multi_A(meshes_A: list[SurfaceMesh],
                             meshes_B: list[SurfaceMesh],
                             meshes_A_winning_rot_vecs: torch.Tensor = None,
                             mesh_A_opt_num_posns: int = 10,
                             mesh_A_fill_soft_mask_holes: bool = True,
                             mesh_B_opt_num_posns: int = 10,
                             mesh_B_opt_posn_trans_bounds: tuple[float, float] = (-0.1, 0.1),
                             mesh_B_opt_use_area_proportion: bool = True,
                             mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                             mesh_B_opt_iters: int = 100,
                             mesh_B_opt_learning_rate: float = 0.05,
                             mesh_A_pso_c1: float = 0.25,
                             mesh_A_pso_c2: float = 0.25,
                             mesh_A_pso_w: float = 0.5,
                             mesh_A_pso_iters: int = 10,
                             proj_A_pso_c1: float = 0.05,
                             proj_A_pso_c2: float = 0.05,
                             proj_A_pso_w: float = 0.5,
                             proj_A_pso_iters: int = 10,
                             pso_verbose: bool = False,
                             image_shape: tuple[int, int] = (256, 256)):
  '''
  Optimize the orientation of a set of A meshes to block a set of B meshes. Specifically, orient
  all the A meshes so that the projection of their union is single silhouette that none of the
  B meshes can be oriented to have their projection contained in it.

  Input:
  - meshes_A: list[kaolin.rep.SurfaceMesh]
      List of meshes to optimize orientations of so that the projection of their union
      blocks all B meshes while minimizing its area.

  - meshes_B: list[kaolin.rep.SurfaceMesh]
      List of meshes to block.

  - meshes_A_winning_rot_vecs: torch.Tensor of shape (a, 6), optional (default None)
      Winning orientations of each A mesh. If they are not provided, they will be computed here.

  - mesh_A_opt_num_posns: int, optional (default 10)
      Number of random positions, or particles, of mesh A to use for PSO.

  - mesh_A_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes of soft mask of mesh A.

  - mesh_B_opt_num_posns: int, optional (default 10)
      Number of randomly generated positions to use whenever optimizing orientation of a B mesh.

  - mesh_B_opt_posn_trans_bounds: tuple[float, float], optional (default (-0.1, 0.1))
      Bounds for translation vectors of the randomly generated positions for B meshes.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - mesh_A_pso_c1: float, optional (default 0.25)
      PSO cognitive coefficient for finding a winning orientation of each mesh A.

  - mesh_A_pso_c2: float, optional (default 0.25)
      PSO social coefficient for finding a winning orientation of each mesh A.

  - mesh_A_pso_w: float, optional (default 0.5)
      PSO inertia weight for finding a winning orientation of each mesh A.

  - mesh_A_pso_iters: int, optional (default 10)
      Number of PSO iterations to run for finding a winning orientation of each mesh A.

  - proj_A_pso_c1: float, optional (default 0.05)
      PSO cognitive coefficient for optimizing a projection of mesh A with that of a base mesh.

  - proj_A_pso_c2: float, optional (default 0.05)
      PSO social coefficient for optimizing a projection of mesh A with that of a base mesh.

  - proj_A_pso_w: float, optional (default 0.5)
      PSO inertia weight for optimizing a projection of mesh A with that of a base mesh.

  - proj_A_pso_iters: int, optional (default 10)
      Number of PSO iterations to run for optimizing a projection of mesh A with that of a base mesh.

  - pso_verbose: bool, optional (default False)
      Option to print logs during PSO.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - best_val: float
      Best objective value achieved.

  - union_mesh: kaolin.rep.SurfaceMesh
      Union of all A meshes in optimized orientations.

  - meshes_A_optimized_rot_vecs: torch.Tensor of shape (a, 6)
      Optimized rotation vector for each A mesh.

  - meshes_A_optimized_trans_vecs: torch.Tensor of shape (a, 2)
      Optimized translation vector for each A mesh.
  '''

  num_meshes_A = len(meshes_A)
  mesh_A_win_threshold = 0.05

  # Find a winning orientation for each A mesh if it was not given as an argument.
  if meshes_A_winning_rot_vecs is None:
    meshes_A_winning_rot_vecs = torch.zeros(num_meshes_A, 6).cuda()

    for i, mesh_A in enumerate(meshes_A):
      best_val, best_posn = optimize_posn_of_A(mesh_A,
                                               meshes_B,
                                               mesh_A_opt_num_posns,
                                               mesh_B_opt_num_posns,
                                               mesh_A_fill_soft_mask_holes,
                                               mesh_B_opt_posn_trans_bounds,
                                               mesh_B_opt_use_area_proportion,
                                               mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                               mesh_B_opt_iters,
                                               mesh_B_opt_learning_rate,
                                               mesh_A_pso_c1,
                                               mesh_A_pso_c2,
                                               mesh_A_pso_w,
                                               mesh_A_pso_iters,
                                               pso_verbose,
                                               image_shape)
      meshes_A_winning_rot_vecs[i] = best_posn

      # If there exists an A mesh without a winning orientation, then the overall set of A meshes
      # cannot win against the set of B meshes.
      if best_val < mesh_A_win_threshold:
        print(f'Warning: Winning orientation may not have been found for mesh A at index {i}.')
  
  # Create a copy of A meshes in their winning orientations.
  meshes_A_rotated = copy.deepcopy(meshes_A)
  for mesh, rot_vec in zip(meshes_A_rotated, meshes_A_winning_rot_vecs):
    mesh.vertices = orie.get_rotated_vtxs(mesh.vertices, rot_vec)

  meshes_A_rotated_soft_masks = torch.stack([get_soft_masks(mesh_A_rotated.vertices.unsqueeze(0),
                                                            mesh_A_rotated.faces,
                                                            image_shape,
                                                            mesh_A_fill_soft_mask_holes)[0]
                                             for mesh_A_rotated in meshes_A_rotated])
  meshes_A_rotated_soft_masks_areas = torch.sum(meshes_A_rotated_soft_masks, dim=(-2, -1))
  meshes_A_sort_idxs = torch.argsort(meshes_A_rotated_soft_masks_areas, descending=True)
  meshes_A_rotated_sorted = [meshes_A_rotated[idx] for idx in meshes_A_sort_idxs]

  # Space out meshes along z-axis so that all mesh unions will operate on disjoint meshes for speed.
  # This is not problematic for later transformations since they will only affect x and y coordinates.
  z_axis_offsets = mesh_utils.get_z_axis_offsets_to_space_out_meshes(meshes_A_rotated_sorted)
  mesh_utils.space_out_meshes(meshes_A_rotated_sorted, z_axis_offsets)

  meshes_A_optimized_rot_vecs = meshes_A_winning_rot_vecs.detach().clone()
  meshes_A_optimized_trans_vecs = torch.zeros(num_meshes_A, 2).cuda()

  curr_base_mesh = meshes_A_rotated_sorted[0]
  for i in range(1, num_meshes_A):
    mesh_A_rotated = meshes_A_rotated_sorted[i]

    best_val, best_posn_angle, best_posn_trans_vec = optimize_winning_posn_of_mesh_A_with_base_mesh(curr_base_mesh,
                                                                                                    mesh_A_rotated,
                                                                                                    meshes_B,
                                                                                                    mesh_A_fill_soft_mask_holes,
                                                                                                    mesh_A_opt_num_posns,
                                                                                                    mesh_A_fill_soft_mask_holes,
                                                                                                    mesh_B_opt_num_posns,
                                                                                                    mesh_B_opt_posn_trans_bounds,
                                                                                                    mesh_B_opt_use_area_proportion,
                                                                                                    mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                                    mesh_B_opt_iters,
                                                                                                    mesh_B_opt_learning_rate,
                                                                                                    proj_A_pso_c1,
                                                                                                    proj_A_pso_c2,
                                                                                                    proj_A_pso_w,
                                                                                                    proj_A_pso_iters,
                                                                                                    pso_verbose,
                                                                                                    image_shape)

    z_axis_rot_mat = torch.eye(3).cuda()
    z_axis_rot_mat[:2,:2] = orie.get_z_axis_rot_mats(best_posn_angle.unsqueeze(0))[0]
    mesh_A_sort_idx = meshes_A_sort_idxs[i]
    mesh_A_optimized_rot_mat = torch.matmul(z_axis_rot_mat, orie.get_rot_mats(meshes_A_winning_rot_vecs[mesh_A_sort_idx]))
    meshes_A_optimized_rot_vecs[mesh_A_sort_idx] = torch.transpose(mesh_A_optimized_rot_mat, 0, 1)[:2,:].flatten()
    meshes_A_optimized_trans_vecs[mesh_A_sort_idx] = best_posn_trans_vec

    mesh_A_optimized_V = mesh_A_rotated.vertices.detach().clone()
    mesh_A_optimized_V[:,:2] = orie.get_transformed_vtxs_2D(mesh_A_rotated.vertices[:,:2],
                                                            best_posn_angle.unsqueeze(0),
                                                            best_posn_trans_vec)
    mesh_A_optimized_V = mesh_A_optimized_V.detach().clone().cpu().numpy()
    mesh_A_optimized_F = mesh_A_rotated.faces.detach().clone().cpu().numpy()
    curr_base_mesh_V = curr_base_mesh.vertices.detach().clone().cpu().numpy()
    curr_base_mesh_F = curr_base_mesh.faces.detach().clone().cpu().numpy()
    next_base_mesh_V, next_base_mesh_F = mesh_boolean(curr_base_mesh_V,
                                                      curr_base_mesh_F,
                                                      mesh_A_optimized_V,
                                                      mesh_A_optimized_F,
                                                      boolean_type='union')
    next_base_mesh = SurfaceMesh(torch.FloatTensor(next_base_mesh_V).cuda(), torch.LongTensor(next_base_mesh_F).cuda())
    curr_base_mesh = next_base_mesh

  union_mesh = curr_base_mesh

  return best_val, union_mesh, meshes_A_optimized_rot_vecs, meshes_A_optimized_trans_vecs


def optimize_posn_of_mesh_B(union_mesh: SurfaceMesh,
                            mesh_B: SurfaceMesh,
                            union_mesh_fill_soft_mask_holes: bool = True,
                            mesh_B_opt_num_posns: int = 10,
                            mesh_B_opt_posn_trans_bounds: tuple[float, float] = (-0.1, 0.1),
                            mesh_B_opt_use_area_proportion: bool = True,
                            mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                            mesh_B_opt_iters: int = 100,
                            mesh_B_opt_learning_rate: float = 0.05,
                            image_shape: tuple[int, int] = (256, 256)):
  '''
  Optimize the orientation of mesh B against the projection of another mesh in its
  optimized orientation to maximize the overlap or projections.

  Input:
  - union_mesh: kaolin.rep.SurfaceMesh
      Union of all A meshes in optimized orientations.

  - mesh_B: kaolin.rep.SurfaceMesh
      Mesh to optimize against projection of the union mesh for overlap.

  - union_mesh_fill_soft_mask_holes: bool, optional (default True)
      Option to fill all holes of soft mask of the union mesh.

  - mesh_B_opt_num_posns: int, optional (default 10)
      Number of randomly generated positions to use whenever optimizing orientation of a B mesh.

  - mesh_B_opt_posn_trans_bounds: tuple[float, float], optional (default (-0.1, 0.1))
      Bounds for translation vectors of the randomly generated positions for B meshes.

  - mesh_B_opt_use_area_proportion: bool, optional (default True)
      Option to return the non-overlapping area divided by the total area
      of the corresponding soft mask of B or just the area as is.

  - mesh_B_opt_vtx_out_of_bounds_loss_weight: float, optional (default 1000.0)
      Value to multiply loss calculated for vertices out of bounds of raster.

  - mesh_B_opt_iters: int, optional (default 100)
      Number of iterations for optimization of orientations of B meshes.

  - mesh_B_opt_learning_rate: float, optional (default 0.05)
      Learning rate for optimization of orientations of B meshes.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of the soft masks.

  Output:
  - best_loss: float

  - best_rot_vec: torch.Tensor of shape (6,)

  - best_trans_vec: torch.Tensor of shape (2,)
  '''

  rot_vec_per_B_posn = orie.generate_rotation_vectors(mesh_B_opt_num_posns)
  trans_vec_per_B_posn = orie.generate_uniform_random_vectors(mesh_B_opt_num_posns, 2, mesh_B_opt_posn_trans_bounds[0], mesh_B_opt_posn_trans_bounds[1])

  rot_vec_per_B_posn_per_A_posn = rot_vec_per_B_posn.unsqueeze(0).repeat([1 for _ in range(len(rot_vec_per_B_posn.shape) + 2)])
  trans_vec_per_B_posn_per_A_posn = trans_vec_per_B_posn.unsqueeze(0).repeat([1 for _ in range(len(trans_vec_per_B_posn.shape) + 2)])

  rot_vec_per_A_posn = torch.tensor([[1., 0., 0., 0., 1., 0.]]).cuda() # identity rotation vector

  _, opt_rot_vec_per_B_posn_per_A_posn, opt_trans_vec_per_B_posn_per_A_posn = optimize_B_posn_per_A_posn_per_B_mesh(rot_vec_per_A_posn=rot_vec_per_A_posn,
                                                                                                                    mesh_A=union_mesh,
                                                                                                                    meshes_B=[mesh_B],
                                                                                                                    rot_vec_per_B_posn_per_A_posn_per_B_mesh=rot_vec_per_B_posn_per_A_posn,
                                                                                                                    trans_vec_per_B_posn_per_A_posn_per_B_mesh=trans_vec_per_B_posn_per_A_posn,
                                                                                                                    mesh_A_fill_soft_mask_holes=union_mesh_fill_soft_mask_holes,
                                                                                                                    mesh_B_opt_use_area_proportion=mesh_B_opt_use_area_proportion,
                                                                                                                    mesh_B_opt_vtx_out_of_bounds_loss_weight=mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                                                    mesh_B_opt_iters=mesh_B_opt_iters,
                                                                                                                    mesh_B_opt_learning_rate=mesh_B_opt_learning_rate,
                                                                                                                    image_shape=image_shape)

  # Compute final loss after last optimization step.
  union_mesh_hard_mask = get_rasters(union_mesh.vertices.unsqueeze(0), union_mesh.faces, image_shape, union_mesh_fill_soft_mask_holes)
  loss_per_B_posn = get_loss_per_B_posn_per_A_posn(union_mesh_hard_mask,
                                                   mesh_B.vertices,
                                                   mesh_B.faces,
                                                   opt_rot_vec_per_B_posn_per_A_posn,
                                                   opt_trans_vec_per_B_posn_per_A_posn,
                                                   mesh_B_opt_use_area_proportion,
                                                   True,
                                                   mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                   image_shape)

  best_loss, best_loss_idx = loss_per_B_posn.flatten(0).min(dim=-1)
  best_loss = best_loss.detach().clone()
  best_rot_vec = opt_rot_vec_per_B_posn_per_A_posn[0, 0, best_loss_idx].detach().clone()
  best_trans_vec = opt_trans_vec_per_B_posn_per_A_posn[0, 0, best_loss_idx].detach().clone()

  return best_loss, best_rot_vec, best_trans_vec
