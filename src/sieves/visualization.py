import cv2 as cv
import kaolin as kal
from kaolin.rep import SurfaceMesh
import numpy as np
import torch
from matplotlib import pyplot as plt

from sieves.colors import colors


def rasterize_mesh(mesh: SurfaceMesh,
                   pixel_color: torch.Tensor = torch.tensor([1.0]).cuda(),
                   fill_holes: bool = False,
                   image_shape: tuple = (256, 256)):
  '''
  Project mesh onto the XY-plane and get raster using Kaolin.

  Input:
  - mesh: kaolin.rep.SurfaceMesh
      Mesh to rasterize.

  - pixel_color: torch.Tensor, optional (default torch.tensor([1.0]))
      Pixel color to use for rasterization.

  - fill_holes: bool, optional (default False)
      Option to fill all holes of raster.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of raster.

  Output:
  - raster: torch.Tensor of shape `image_shape`
      Raster of mesh.
  '''

  num_faces = mesh.faces.shape[0]

  face_vertices_z = torch.zeros(1, num_faces, 3).cuda()
  face_vertices_image = mesh.vertices[:,:2][mesh.faces].unsqueeze(0)
  face_features = torch.tile(torch.tensor([1.0]), (1, num_faces, 3, 1)).cuda()

  rendered_features, _ = kal.render.mesh.rasterize(image_shape[0], image_shape[1], face_vertices_z, face_vertices_image, face_features)
  binary_raster = rendered_features[0]
  binary_raster = torch.clamp(binary_raster, 0., 1.)
  binary_raster[binary_raster > 0.] = 1.
  binary_raster = binary_raster.type(torch.IntTensor).cuda()

  if fill_holes:
    binary_raster_np = binary_raster.detach().clone().cpu().numpy().astype(np.uint8)

    # Get the outermost contours.
    contours, _ = cv.findContours(binary_raster_np, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Get raster with all holes filled by filling in the outermost contours.
    filled_raster_np = np.zeros_like(binary_raster_np)
    ctr_idx = -1
    thickness = -1
    fill_pixel_color = (1)
    cv.drawContours(filled_raster_np, contours, ctr_idx, fill_pixel_color, thickness)

    binary_raster = torch.IntTensor(filled_raster_np).cuda()

  raster = binary_raster * pixel_color

  return raster


def show_raster(mesh: SurfaceMesh,
                pixel_color: torch.Tensor = torch.tensor([1.0]).cuda(),
                fill_holes: bool = False,
                image_shape: tuple = (256, 256)):
  '''
  Rasterize a mesh by projecting onto the XY-plane and display it.

  Input:
  - mesh: kaolin.rep.SurfaceMesh
      Mesh to rasterize.

  - pixel_color: torch.Tensor, optional (default torch.tensor([1.0]))
      Pixel color to use for rasterization.

  - fill_holes: bool, optional (default False)
      Option to fill all holes of raster.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution of raster.
  '''

  raster = rasterize_mesh(mesh, pixel_color, fill_holes, image_shape)

  plt.figure()
  plt.imshow(raster.detach().clone().cpu().numpy())

  return


def get_figure_raster(union_mesh: SurfaceMesh,
                      mesh: SurfaceMesh,
                      pixel_color_mesh: torch.Tensor,
                      pixel_color_background: torch.Tensor = torch.IntTensor(colors['sieve']).cuda(),
                      mesh_opacity: float = 0.7,
                      outline_thickness: int = 5,
                      image_shape: tuple[int, int] = (2048, 2048)):
  '''
  Generate raster of projections of meshes A and B for figures.

  Input:
  - union_mesh: kaolin.rep.SurfaceMesh
      Mesh whose projection is the sieve hole.

  - mesh: kaolin.rep.SurfaceMesh
      A mesh in its transformed orientation over the sieve hole.

  - pixel_color_mesh: torch.Tensor of shape (3,)
      Pixel color for the projection of the mesh.

  - pixel_color_background: torch.Tensor of shape (3,), optional (default torch.IntTensor([240, 240, 240]))
      Pixel color for the background, should be same as that of the sieve.

  - mesh_opacity: float, optional (default 0.7)
      Opacity of pixel color for the projection of the mesh.

  - image_shape: tuple[int, int], optional (default (2048, 2048))
      Raster resolution.

  Output:
  - overlaid_raster: numpy array of shape (image_shape[0], image_shape[1], 3)
      Raster for figures with outline for hole.
  '''

  binary_pixel_color = torch.IntTensor([1]).cuda()
  binary_raster_mesh = rasterize_mesh(mesh=mesh,
                                      fill_holes=False,
                                      pixel_color=binary_pixel_color,
                                      image_shape=image_shape)
  raster_mesh = binary_raster_mesh * pixel_color_mesh + (1. - binary_raster_mesh) * pixel_color_background
  raster_mesh_np = raster_mesh.detach().clone().cpu().numpy().astype(np.uint8)

  binary_raster_sieve_hole = rasterize_mesh(mesh=union_mesh,
                                            fill_holes=True,
                                            pixel_color=binary_pixel_color,
                                            image_shape=image_shape)
  pixel_color_white = torch.IntTensor([255, 255, 255]).cuda()
  raster_sieve_hole = binary_raster_sieve_hole * pixel_color_white + (1. - binary_raster_sieve_hole) * pixel_color_background

  contours, _ = cv.findContours(binary_raster_sieve_hole.detach().clone().cpu().numpy().astype(np.uint8),
                                cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_NONE)
  outlined_raster_np = raster_sieve_hole.detach().clone().cpu().numpy().astype(np.uint8)
  ctr_idx = -1
  fill_pixel_color = (0, 0, 0)
  cv.drawContours(outlined_raster_np, contours, ctr_idx, fill_pixel_color, outline_thickness)

  alpha = mesh_opacity * binary_raster_mesh.detach().clone().cpu().numpy()
  overlaid_raster = (alpha * raster_mesh_np + (1. - alpha) * outlined_raster_np).astype(np.uint8)

  return overlaid_raster
