import gpytoolbox as gp
from gpytoolbox.copyleft import mesh_boolean
import kaolin
import numpy as np
import os
import torch

from sieves import mesh_utils
from sieves import visualization as vis


def get_raster_mesh(mesh: kaolin.rep.SurfaceMesh,
                    fill_holes: bool,
                    image_shape: tuple[int, int]):
  '''
  Get a mesh of the raster of the orthographic projection of an input mesh.

  Input:
  - mesh: kaolin.rep.SurfaceMesh
      Mesh to use to create sieve. Should already be in desired orientation.

  - fill_holes: bool
      Option to fill all holes of projection of mesh used for creating sieve hole.

  - image_shape: tuple[int, int]
      Resolution to use for rasters.

  Output:
  - raster_V: numpy array of shape (V, 3)
      Raster mesh vertices.

  - raster_F: numpy array of shape (F, 3)
      Raster mesh faces.
  '''

  raster = vis.rasterize_mesh(mesh, fill_holes=fill_holes, image_shape=image_shape).flatten(-2)
  raster = torch.transpose(raster, 0, 1)
  raster = torch.flip(raster, [1])

  stack_height = 2
  raster_stacked = raster.repeat(stack_height, 1, 1)

  pad_size = 1
  raster_stacked_padded = torch.nn.functional.pad(raster_stacked,
                                                  (0, 0, 0, 0, pad_size, pad_size),
                                                  mode='constant',
                                                  value=0)

  raster_stacked_padded_rescaled = 2 * raster_stacked_padded - 1

  # Get scalar values for marching cubes.
  S = raster_stacked_padded_rescaled.permute(2, 1, 0).flatten().detach().clone().cpu().numpy()

  # Get a grid of vertices of a uniform cube.
  grid_V, _ = gp.regular_cube_mesh(image_shape[0], image_shape[1], stack_height + 2*pad_size)

  # Center the cube.
  grid_V -= 0.5 * np.ones(3)

  # Scale the grid since raster is in [-1, 1] x [-1, 1] and the grid is for a unit cube.
  grid_V[:,0] *= 2.
  grid_V[:,1] *= 2.

  # Get 3D mesh of raster using marching cubes.
  raster_V, raster_F = gp.marching_cubes(S, grid_V, stack_height + 2*pad_size, image_shape[1], image_shape[0])
  raster_F = np.flip(raster_F, axis=1)

  return raster_V, raster_F


def get_plate_mesh(x_scale: float = 4.0,
                   y_scale: float = 4.0,
                   z_scale: float = 0.1):
  '''
  Get rectangular prism plate mesh for fabricating sieve hole by scaling
  a mesh of a unit cube.

  Input:
  - x_scale: float, optional (default 4.0)
      Scale to apply to X-coordinates of unit cube mesh.

  - y_scale: float, optional (default 4.0)
      Scale to apply to Y-coordinates of unit cube mesh.

  - z_scale: float, optional (default 0.1)
      Scale to apply to Z-coordinates of unit cube mesh.

  Output:
  - plate_mesh: kaolin.rep.SurfaceMesh
      Scaled unit cube mesh.
  '''

  filepath_cube_obj = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data', 'cube.obj')
  plate_mesh = mesh_utils.import_mesh(filepath_cube_obj, simulation_scale=1., reader='Python')
  plate_mesh.vertices[:,0] *= x_scale
  plate_mesh.vertices[:,1] *= y_scale
  plate_mesh.vertices[:,2] *= z_scale

  return plate_mesh


def get_sieve_plate_minus_mesh(mesh: kaolin.rep.SurfaceMesh,
                               scale_factor: float,
                               fill_holes: bool = True,
                               sieve_side_length: float = 80.,
                               sieve_thickness: float = 10.,
                               image_shape: tuple[int, int] = (256, 256)):
  '''
  Get plate mesh minus the mesh of the rasterized projection of the input mesh.

  Input:
  - mesh: kaolin.rep.SurfaceMesh
      Mesh to use to create sieve. Should already be in desired orientation.

  - scale_factor: float
      Scale to apply to result to account for the fact that mesh may have been
      simulated at a different scale than 1.0.

  - fill_holes: bool, optional (default True)
      Option to fill all holes of projection of mesh used for creating sieve hole.

  - sieve_side_length: float, optional (default 80.0)
      Side length of sieve mesh.

  - sieve_thickness: float, optional (default 10.0)
      Thickness of sieve mesh.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution to use for rasters.

  Output:
  - sieve_mesh_V: numpy array of shape (V, 3)
      Sieve mesh vertices.

  - sieve_mesh_F: numpy array of shape (F, 3)
      Sieve mesh faces.
  '''

  raster_V, raster_F = get_raster_mesh(mesh, fill_holes, image_shape)
  raster_V[:,0] *= scale_factor
  raster_V[:,1] *= scale_factor

  plate_mesh = get_plate_mesh(x_scale=sieve_side_length, y_scale=sieve_side_length, z_scale=0.1)
  plate_mesh_V = plate_mesh.vertices.detach().clone().cpu().numpy()
  plate_mesh_F = plate_mesh.faces.detach().clone().cpu().numpy()

  sieve_mesh_V, sieve_mesh_F = mesh_boolean(plate_mesh_V,
                                            plate_mesh_F,
                                            raster_V,
                                            raster_F,
                                            boolean_type='difference')
  sieve_mesh_V[:,2] *= 10 * sieve_thickness

  return sieve_mesh_V, sieve_mesh_F


def get_sieve_mesh(mesh_filepath: str,
                   mesh_rot_vec: torch.Tensor,
                   fabrication_scale: float = 1.0,
                   fabrication_offset: float = 1.0,
                   simulation_scale: float = 0.02,
                   fill_holes: bool = True,
                   sieve_side_length: float = 80.0,
                   sieve_thickness: float = 10.0,
                   image_shape: tuple[int, int] = (256, 256)):
  '''
  Get plate mesh with sieve hole being the projection of a mesh.

  Input:
  - mesh_filepath: str
      Filepath of mesh to use for sieve hole.

  - mesh_rot_vec: torch.Tensor of shape (6,)
      Rotation vector for mesh to get desired projection to use for sieve hole.

  - fabrication_scale: float, optional (default 1.0)
      Fabrication scale for mesh.

  - fabrication_offset: float, optional (default 1.0)
      Fabrication offset to apply to mesh.

  - simulation_scale: float, optional (default 0.02)
      Scale to apply to mesh for rasterization.

  - fill_holes: bool, optional (default True)
      Option to fill all holes of projection of mesh used for creating sieve hole.

  - sieve_side_length: float, optional (default 80.0)
      Side length of sieve mesh.

  - sieve_thickness: float, optional (default 10.0)
      Thickness of sieve mesh.

  - image_shape: tuple[int, int], optional (default (256, 256))
      Resolution to use for rasters.

  Output:
  - sieve_mesh_V: numpy array of shape (V, 3)
      Sieve mesh vertices.

  - sieve_mesh_F: numpy array of shape (F, 3)
      Sieve mesh faces.
  '''

  mesh = mesh_utils.import_mesh(mesh_filepath,
                                fabrication_scale=fabrication_scale,
                                fabrication_offset=fabrication_offset,
                                simulation_normalize=False,
                                simulation_scale=simulation_scale)
  mesh_transformed = mesh_utils.get_transformed_mesh(mesh, mesh_rot_vec)

  mesh_scale_factor = 1. / simulation_scale
  sieve_mesh_V, sieve_mesh_F = get_sieve_plate_minus_mesh(mesh_transformed,
                                                          mesh_scale_factor,
                                                          fill_holes,
                                                          sieve_side_length,
                                                          sieve_thickness,
                                                          image_shape)

  return sieve_mesh_V, sieve_mesh_F
