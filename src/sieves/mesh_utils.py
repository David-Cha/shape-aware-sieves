import copy
import gpytoolbox as gp
from gpytoolbox.copyleft import mesh_boolean
import igl
import kaolin as kal
from kaolin.rep import SurfaceMesh
import numpy.typing as npt
import os
import torch

from sieves import orientation as orie


def import_mesh(filepath: str,
                fabrication_scale: float = 1.,
                fabrication_offset: float = None,
                simulation_normalize: bool = False,
                simulation_scale: float = 1.,
                reader: str = None):
  '''
  Import OBJ or STL file into Kaolin mesh format.

  Input:
  - filepath: str
      Path to OBJ or STL file.

  - fabrication_scale: float, optional (default 1.0)
      Factor to scale mesh vertex positions by for actual fabricated size.

  - fabrication_offset: float, optional (default None)
      Offset for dilating mesh for having a margin of error for phyiscally fabricated results.

  - simulation_normalize: bool, optional (default False)
      Option to normalize mesh vertex positions before scaling by `simulation_scale`.

  - simulation_scale: float, optional (default 1.0)
      Factor to scale mesh vertex positions by for optimization since meshes that are too large
      will not fit in rasters used during optimization.

  - reader: str, optional (default None)
      Reader to use for importing mesh via gpytoolbox (C++ or Python).

  Output:
  - mesh: kaolin.rep.SurfaceMesh
      Imported mesh.
  '''

  V, F = gp.read_mesh(filepath, reader=reader)
  mesh = SurfaceMesh(torch.FloatTensor(V), torch.LongTensor(F)).cuda()
  mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0)).squeeze(0)

  # Apply fabrication scales first. Note that scaling should occur before offsetting.
  mesh.vertices *= fabrication_scale

  if fabrication_offset is not None:
    assert(isinstance(fabrication_offset, float))
    offset_mesh_V, offset_mesh_F = gp.offset_surface(mesh.vertices.clone().detach().cpu().numpy(),
                                                     mesh.faces.clone().detach().cpu().numpy(),
                                                     fabrication_offset,
                                                     100)
    mesh = SurfaceMesh(torch.FloatTensor(offset_mesh_V), torch.LongTensor(offset_mesh_F)).cuda()

  # Then apply scales to be used during simulation. Note that normalization should occur before scaling.
  if simulation_normalize:
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

  mesh.vertices *= simulation_scale

  return mesh


def import_mesh_from_config(mesh_config: dict,
                            data_dir: str,
                            simulation_scale: float):
  '''
  Import mesh based on configuration file.

  Input:
  - mesh_config: dict
      Configuration dictionary for mesh.

  - data_dir: str
      Path to directory where all mesh files are stored.

  - simulation_scale: float
      Factor to scale mesh vertex positions by for optimization since meshes that are too large
      will not fit in rasters used during optimization.

  Output:
  - mesh: kaolin.rep.SurfaceMesh
      Imported mesh.
  '''

  mesh_filepath = os.path.join(data_dir, mesh_config['filename'])
  mesh = import_mesh(filepath=mesh_filepath,
                     fabrication_scale=mesh_config['fabrication_scale'],
                     fabrication_offset=mesh_config['fabrication_offset'],
                     simulation_normalize=False,
                     simulation_scale=simulation_scale)

  return mesh


def export_mesh_as_stl(filename: str,
                       mesh_V: npt.NDArray,
                       mesh_F: npt.NDArray):
  '''
  Save mesh as STL file for 3D printing.

  Input:
  - filename: str
      Filename for saving mesh, must end with '.stl'.

  - mesh_V: numpy array of shape (V, 3)
      Mesh vertices.

  - mesh_F: numpy array of shape (F, 3)
      Mesh faces.

  Output:
  - success: bool
      Bool indicating if write succeeded.
  '''

  return igl.write_triangle_mesh(filename, mesh_V, mesh_F)


def get_transformed_mesh(mesh: SurfaceMesh,
                         rot_vec: torch.Tensor,
                         trans_vec: torch.Tensor = torch.zeros(2).cuda()):
  '''
  Get mesh transformed by applying a rotation and then a translation.

  Input:
  - mesh: kaolin.rep.SurfaceMesh
      Mesh to transform.

  - rot_vec: torch.Tensor of shape (6,)
      6D vector to rotate mesh vertices by.

  - trans_vec: torch.Tensor of shape (2,), optional (default torch.tensor([0.0, 0.0]))
      2D vector to translate X and Y coordinates of mesh vertices.

  Output:
  - transformed_mesh: kaolin.rep.SurfaceMesh
      Resulting mesh after transformations.
  '''

  transformed_mesh_vtxs = orie.get_transformed_vtxs(mesh.vertices, rot_vec.unsqueeze(0), trans_vec.unsqueeze(0))[0]
  transformed_mesh = SurfaceMesh(transformed_mesh_vtxs, mesh.faces)

  return transformed_mesh


def get_aabb_corners(mesh: SurfaceMesh):
  '''
  Get the two AABB corners of a mesh.

  Input:
  - mesh: kaolin.rep.SurfaceMesh
      Mesh to get AABB corners of.

  Output:
  - aabb_min: torch.Tensor of shape (3,)
      Lower AABB corner of mesh.

  - aabb_max: torch.Tensor of shape (3,)
      Upper AABB corner of mesh.
  '''

  vtxs = mesh.vertices
  aabb_min = torch.min(vtxs.T, 1).values
  aabb_max = torch.max(vtxs.T, 1).values

  return aabb_min, aabb_max


def get_z_axis_offsets_to_space_out_meshes(meshes: list[SurfaceMesh]):
  '''
  Given a list of meshes, find a set of values to translate the meshes along the z-axis by
  so that under any rotation and xy-translation applied to the meshes, none will intersect.

  Input:
  - meshes: list[kaolin.rep.SurfaceMesh]
      List of meshes.

  Output:
  - offsets: torch.Tensor of shape (n,) where n = len(meshes)
      Offsets to apply to z-coordinates for each mesh.
  '''

  curr_offset = 0.
  prev_radius = 0.
  curr_radius = 0.
  offsets = torch.zeros(len(meshes)).cuda()

  for i, mesh in enumerate(meshes):
    aabb_min, aabb_max = get_aabb_corners(mesh)

    prev_radius = curr_radius
    curr_radius = 0.5 * torch.norm(aabb_max - aabb_min)
    curr_offset += (prev_radius + curr_radius)

    offsets[i] = curr_offset

  return offsets


def space_out_meshes(meshes: list[SurfaceMesh],
                     offsets: torch.Tensor):
  '''
  Translate a list of meshes along the z-axis so that for any rotations and xy-translations
  applied to the meshes, none will intersect.

  Input:
  - meshes: list[kaolin.rep.SurfaceMesh]
      List of meshes.

  - offsets: torch.Tensor of shape (n,) where n = len(meshes)
      Offsets to apply to z-coordinates for each mesh.
  '''

  for i, mesh in enumerate(meshes):
    mesh.vertices[:, 2] += offsets[i]

  return


def get_union_mesh(meshes: list[SurfaceMesh]):
  '''
  Get union of a list of meshes that have been offset along the z-axis so that they are
  not intersecting.

  Input:
  - meshes: list[kaolin.rep.SurfaceMesh]
      List of meshes.

  Output:
  - union_mesh: kaolin.rep.SurfaceMesh
      Union of meshes.
  '''

  meshes_copy = copy.deepcopy(meshes)
  offsets = get_z_axis_offsets_to_space_out_meshes(meshes_copy)
  space_out_meshes(meshes_copy, offsets)

  curr_base_mesh = meshes_copy[0]

  for i in range(1, len(meshes_copy)):
    curr_mesh_A = meshes_copy[i]
    curr_base_mesh_V = curr_base_mesh.vertices.detach().clone().cpu().numpy()
    curr_base_mesh_F = curr_base_mesh.faces.detach().clone().cpu().numpy()
    next_base_mesh_V, next_base_mesh_F = mesh_boolean(curr_base_mesh_V,
                                                      curr_base_mesh_F,
                                                      curr_mesh_A.vertices.detach().clone().cpu().numpy(),
                                                      curr_mesh_A.faces.detach().clone().cpu().numpy(),
                                                      boolean_type='union')
    next_base_mesh = SurfaceMesh(torch.FloatTensor(next_base_mesh_V).cuda(), torch.LongTensor(next_base_mesh_F).cuda())
    curr_base_mesh = next_base_mesh

  union_mesh = curr_base_mesh

  return union_mesh
