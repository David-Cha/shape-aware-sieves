import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
RESULTS_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'results'))
SCRIPTS_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'scripts'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
import blendertoolbox as bt
import bmesh
import bpy
import gpytoolbox as gp
from gpytoolbox.copyleft import mesh_boolean
import igl
import json
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import torch

from sieves import mesh_utils
from sieves import visualization as vis
from sieves.colors import colors


'''
Setup color objects for Blender.
'''

color_A = colors['A']
mesh_color_A_rgba = (color_A[0] / 255., color_A[1] / 255., color_A[2] / 255., 1.0)
mesh_color_A = bt.colorObj(mesh_color_A_rgba, 0.5, 1.0, 1.0, 0.0, 0.0)

color_B = colors['B']
mesh_color_B_rgba = (color_B[0] / 255., color_B[1] / 255., color_B[2] / 255., 1.0)
mesh_color_B = bt.colorObj(mesh_color_B_rgba, 0.5, 2.0, 1.0, 0.0, 0.0)

color_sieve = colors['sieve']
sieve_color_rgba = (color_sieve[0] / 255., color_sieve[1] / 255., color_sieve[2] / 255., 1.0)
sieve_color = bt.colorObj(sieve_color_rgba, 0.5, 1.0, 1.0, 0.0, 0.0)


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('--resx', type=int, default=2048)
  parser.add_argument('--resy', type=int, default=2048)
  parser.add_argument('--samples', type=int, default=200)
  args = parser.parse_args()

  return args


def process_config_and_result_dicts(result_name: str):
  '''
  Read the configuration and results JSON files.
  '''

  # Read the configuration file.
  config_filepath = os.path.abspath(os.path.join(SCRIPTS_DIR, f'{result_name}.json'))
  with open(config_filepath, 'r') as config_file:
    config_dict = json.load(config_file)

  # Read the results file.
  result_dir = os.path.abspath(os.path.join(RESULTS_DIR, result_name))
  result_json_filepath = os.path.abspath(os.path.join(result_dir, 'results.json'))
  with open(result_json_filepath, 'r') as result_json_file:
    result_dict = json.load(result_json_file)
  
  return config_dict, result_dict


def init_blender(res: tuple[int, int],
                 num_samples: int,
                 exposure: float = 1.5):
  '''
  Initialize Blender.
  '''

  bt.blenderInit(res[0], res[1], num_samples, exposure)

  return


def split_edges(mesh_object: object):
  '''
  Split Blender mesh edges using default settings.
  '''

  bm = bmesh.new()
  mesh_data = mesh_object.data
  bm.from_mesh(mesh_data)
  bmesh.ops.split_edges(bm, edges=bm.edges)
  bm.to_mesh(mesh_data)
  bm.clear() 
  bm.free() 

  return


def smooth_mesh(mesh_V: np.ndarray,
                mesh_F: np.ndarray,
                t: float = 0.1,
                iters: int = 3):
  '''
  Apply Laplacian smoothing to mesh.
  '''

  mesh_V_smoothed = np.copy(mesh_V)

  for _ in range(iters):
    L = gp.cotangent_laplacian(mesh_V_smoothed, mesh_F)
    M = gp.massmatrix(mesh_V_smoothed, mesh_F)

    A = M + t * L
    b = M * mesh_V_smoothed
    mesh_V_smoothed = sp.sparse.linalg.spsolve(A, b)

  return mesh_V_smoothed, mesh_F


def render_mesh(img_filepath: str,
                blend_filepath: str,
                mesh: object,
                mesh_color: bt.colorObj,
                use_edge_splitting: bool,
                use_smooth_shading: bool,
                shadow_brightness: float,
                light_sun_angle: tuple[float, float, float],
                light_sun_strength: int,
                light_sun_shadow_softness: float,
                light_ambient_color: tuple[float, float, float, float],
                cam: object):
  '''
  Render a mesh for figures.
  '''

  mesh_object = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'][0]

  # Split edges.
  if use_edge_splitting:
    split_edges(mesh_object)

  # Set shading.
  if use_smooth_shading:
    with bpy.context.temp_override(selected_editable_objects=[mesh_object]):
      bpy.ops.object.shade_smooth()

  # Set material.
  bt.setMat_plastic(mesh, mesh_color)

  # Set invisible plane (shadow catcher).
  if shadow_brightness is not None:
    bt.invisibleGround(shadowBrightness=shadow_brightness)

  # Set sun light.
  if light_sun_strength is not None:
    _sun = bt.setLight_sun(light_sun_angle, light_sun_strength, light_sun_shadow_softness)

  # Set ambient light.
  if light_ambient_color is not None:
    bt.setLight_ambient(color=light_ambient_color) 

  # Set gray shadow to completely white with a threshold.
  bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

  # Save Blender file so that you can adjust parameters in the UI.
  if blend_filepath is not None:
    bpy.ops.wm.save_mainfile(filepath=blend_filepath)

  # Save render.
  bt.renderImage(img_filepath, cam)

  return


def render_sieve_mesh(result_name: str,
                      sieve_pos: tuple[float, float, float] = (0.2, 0.0, 0.3),
                      sieve_rot: tuple[float, float, float] = (0, 0, 45),
                      sieve_scale: float = 0.01,
                      cam_location: tuple[float, float, float] = (1., 0., 3.),
                      cam_rotation: tuple[float, float, float] = (15, 0, 90),
                      focal_length: int = 45):
  '''
  Render a sieve mesh for figures.
  '''

  # Import mesh.
  result_dir = os.path.abspath(os.path.join(RESULTS_DIR, result_name))
  sieve_filepath = os.path.abspath(os.path.join(result_dir, 'sieve.stl'))
  sieve_V, sieve_F = gp.read_mesh(sieve_filepath)
  sieve_V_smoothed, sieve_F_smoothed = smooth_mesh(sieve_V, sieve_F)
  sieve_scale_xyz = (sieve_scale, sieve_scale, sieve_scale)
  sieve_mesh = bt.readNumpyMesh(sieve_V_smoothed, sieve_F_smoothed, sieve_pos, sieve_rot, sieve_scale_xyz)

  # Set color.
  color_sieve = colors['sieve']
  sieve_color_rgba = (color_sieve[0] / 255., color_sieve[1] / 255., color_sieve[2] / 255., 1.0)
  mesh_color = bt.colorObj(sieve_color_rgba, 0.5, 1.0, 1.0, 0.0, 0.0) # colorObj(RGBA, H, S, V, Bright, Contrast)

  # Set lighting and shadows.
  shadow_brightness = None # no ground shadow for sieves
  light_sun_angle = (6, -30, -155)
  light_sun_strength = 4
  light_sun_shadow_softness = 0.3
  light_ambient_color = (0.3, 0.3, 0.3, 1.0)

  # Set camera (ideally change mesh instead of camera, unless you want to adjust the elevation).
  cam = bt.setCamera_from_UI(cam_location, cam_rotation, focal_length)

  # Setup output filepaths.
  img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_sieve.png'))
  blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_sieve.blend'))

  render_mesh(img_filepath=img_filepath,
              blend_filepath=blend_filepath,
              mesh=sieve_mesh,
              mesh_color=mesh_color,
              use_edge_splitting=True,
              use_smooth_shading=True,
              shadow_brightness=shadow_brightness,
              light_sun_angle=light_sun_angle,
              light_sun_strength=light_sun_strength,
              light_sun_shadow_softness=light_sun_shadow_softness,
              light_ambient_color=light_ambient_color,
              cam=cam)
  
  return


def render_player_mesh(result_name: str,
                       mesh_filename: str,
                       mesh_fab_scale: float,
                       mesh_color: bt.colorObj,
                       mesh_pos: tuple[float, float, float],
                       mesh_rot: tuple[float, float, float],
                       mesh_scale: float = 0.01,
                       use_edge_splitting: bool = False,
                       use_smooth_shading: bool = True,
                       cam_location: tuple[float, float, float] = (4., 0., 2.),
                       cam_rotation: tuple[float, float, float] = (60, 0, 90),
                       focal_length: int = 45,
                       is_A: bool = None):
  '''
  Render a mesh.
  '''

  # Import mesh.
  mesh_filepath = os.path.abspath(os.path.join(DATA_DIR, mesh_filename))
  mesh_kaolin = mesh_utils.import_mesh(mesh_filepath, mesh_fab_scale)
  mesh_V = mesh_kaolin.vertices.detach().clone().cpu().numpy()
  mesh_F = mesh_kaolin.faces.detach().clone().cpu().numpy()
  mesh_scale_xyz = (mesh_scale, mesh_scale, mesh_scale)
  mesh = bt.readNumpyMesh(mesh_V,
                          mesh_F,
                          mesh_pos,
                          mesh_rot,
                          mesh_scale_xyz)

  # Set lighting and shadows.
  shadow_brightness = 0.9
  light_sun_angle = (6, -30, -155)
  light_sun_strength = 4
  light_sun_shadow_softness = 0.3
  light_ambient_color = (0.3, 0.3, 0.3, 1.0)

  # Set camera (ideally change mesh instead of camera, unless you want to adjust the elevation).
  cam = bt.setCamera_from_UI(cam_location, cam_rotation, focal_length)

  # Set output filepaths.
  mesh_name = mesh_filename.rsplit('.', 1)[0]

  if is_A is not None:
    player = 'A' if is_A else 'B'
    blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{player}_{mesh_name}.blend'))
    img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{player}_{mesh_name}.png'))
  else:
    blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{mesh_name}.blend'))
    img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{mesh_name}.png'))

  render_mesh(img_filepath=img_filepath,
              blend_filepath=blend_filepath,
              mesh=mesh,
              mesh_color=mesh_color,
              use_edge_splitting=use_edge_splitting,
              use_smooth_shading=use_smooth_shading,
              shadow_brightness=shadow_brightness,
              light_sun_angle=light_sun_angle,
              light_sun_strength=light_sun_strength,
              light_sun_shadow_softness=light_sun_shadow_softness,
              light_ambient_color=light_ambient_color,
              cam=cam)

  return


def render_player_meshes(result_name: str,
                         mesh_configs: list[dict],
                         mesh_color: bt.colorObj,
                         mesh_posns: list[tuple[float, float, float]],
                         mesh_rots: list[tuple[float, float, float]],
                         mesh_scale: float = 0.01,
                         use_edge_splitting_flags: list[bool] = None,
                         use_smooth_shading_flags: list[bool] = None,
                         cam_location: tuple[float, float, float] = (4., 0., 2.),
                         cam_rotation: tuple[float, float, float] = (60, 0, 90),
                         focal_length: int = 45,
                         res: tuple[int, int] = None,
                         num_samples: int = None,
                         is_A: bool = None):
  '''
  Render a set of meshes.
  '''

  if use_edge_splitting_flags is None:
    use_edge_splitting_flags = [False for _ in mesh_configs]

  if use_smooth_shading_flags is None:
    use_smooth_shading_flags = [True for _ in mesh_configs]

  for idx, mesh_config in enumerate(mesh_configs):
    init_blender(res, num_samples)
    render_player_mesh(result_name=result_name,
                       mesh_filename=mesh_config['filename'],
                       mesh_fab_scale=mesh_config['fabrication_scale'],
                       mesh_color=mesh_color,
                       mesh_pos=mesh_posns[idx],
                       mesh_rot=mesh_rots[idx],
                       mesh_scale=mesh_scale,
                       use_edge_splitting=use_edge_splitting_flags[idx],
                       use_smooth_shading=use_smooth_shading_flags[idx],
                       cam_location=cam_location,
                       cam_rotation=cam_rotation,
                       focal_length=focal_length,
                       is_A=is_A)

  return


def render_mesh_orthographic(result_name: str,
                             mesh_filename: str,
                             mesh_color: bt.colorObj,
                             mesh_rot_vec: tuple[float, float, float, float, float, float],
                             mesh_pos: tuple[float, float, float] = (0, 0, 0),
                             mesh_rot: tuple[float, float, float] = (0, 0, 0),
                             mesh_fab_scale: float = 1.0,
                             mesh_scale: float = 0.05,
                             use_edge_splitting: bool = False,
                             use_smooth_shading: bool = True,
                             cam_location: tuple[float, float, float] = (0., 0., 6.),
                             is_A: bool = None):
  '''
  Render orthographic view of mesh in optimized orientation.
  '''

  # Import mesh.
  mesh_filepath = os.path.abspath(os.path.join(DATA_DIR, mesh_filename))
  mesh_kaolin = mesh_utils.import_mesh(mesh_filepath, mesh_fab_scale)
  mesh_transformed = mesh_utils.get_transformed_mesh(mesh_kaolin, torch.tensor(mesh_rot_vec).cuda())
  mesh_transformed_V = mesh_transformed.vertices.detach().clone().cpu().numpy()
  mesh_transformed_F = mesh_transformed.faces.detach().clone().cpu().numpy()
  mesh_scale_xyz = (mesh_scale, mesh_scale, mesh_scale)
  mesh = bt.readNumpyMesh(mesh_transformed_V,
                          mesh_transformed_F,
                          mesh_pos,
                          mesh_rot,
                          mesh_scale_xyz)

  # Set lighting and shadows.
  shadow_brightness = None # no ground shadow for othographic views
  light_sun_angle = (6, -30, -155)
  light_sun_strength = 4
  light_sun_shadow_softness = 0.3
  light_ambient_color = (0.3, 0.3, 0.3, 1.0)

  # Set orthographic camera.
  cam = bt.setCamera_from_UI(cam_location, (0, 0, 0), 45)
  cam.data.type = 'ORTHO'

  # Set output filepaths.
  mesh_name = mesh_filename.rsplit('.', 1)[0]

  if is_A is not None:
    player = 'A' if is_A else 'B'
    blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_ortho_mesh_{player}_{mesh_name}.blend'))
    img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_ortho_mesh_{player}_{mesh_name}.png'))
  else:
    blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_ortho_mesh_{mesh_name}.blend'))
    img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_ortho_mesh_{mesh_name}.png'))

  render_mesh(img_filepath=img_filepath,
              blend_filepath=blend_filepath,
              mesh=mesh,
              mesh_color=mesh_color,
              use_edge_splitting=use_edge_splitting,
              use_smooth_shading=use_smooth_shading,
              shadow_brightness=shadow_brightness,
              light_sun_angle=light_sun_angle,
              light_sun_strength=light_sun_strength,
              light_sun_shadow_softness=light_sun_shadow_softness,
              light_ambient_color=light_ambient_color,
              cam=cam)

  return


def compute_offset_for_mesh_in_sieve(mesh_V: np.ndarray,
                                     mesh_F: np.ndarray,
                                     sieve_V: np.ndarray,
                                     sieve_F: np.ndarray,
                                     mesh_blocked: bool):
  '''
  Compute minimum z-axis offset needed for rendering a mesh blocked by a sieve to prevent intersection.
  If the mesh is admitted by the sieve, then just compute an offset so that it is near the top of the sieve.
  '''

  mesh_z_min = np.min(mesh_V, axis=0)[2]
  mesh_z_max = np.max(mesh_V, axis=0)[2]
  sieve_z_min = np.min(sieve_V, axis=0)[2]
  sieve_z_max = np.max(sieve_V, axis=0)[2]

  mesh_is_pwn = igl.piecewise_constant_winding_number(mesh_F)

  if not mesh_is_pwn:
    # If mesh is not PWN, then mesh boolean operations return incorrect results,
    # so just compute a valid but rough z-offset.

    if mesh_blocked:
      tol = 5.
      mesh_z_offset = (sieve_z_max + tol) - mesh_z_min
    else:
      tol = 5.
      mesh_z_offset = (sieve_z_max + tol) - mesh_z_max

  else:
    # If mesh is PWN, we can use mesh boolean operations to compute a z-offset
    # so that the mesh is placed inside the sieve hole as much as possible.

    # Scale and translate sieve mesh so that it bounds the player mesh along the z-axis.
    tolerance = 5.
    sieve_z_scale = (mesh_z_max - mesh_z_min + 2 * tolerance) / (sieve_z_max - sieve_z_min)
    sieve_V_transformed = sieve_V.copy()

    if sieve_z_scale > 1.:
      sieve_V_transformed[:,2] *= sieve_z_scale

    sieve_z_translation = (mesh_z_min - tolerance) - np.min(sieve_V_transformed, axis=0)[2]
    sieve_V_transformed[:,2] += sieve_z_translation

    # Get intersection of sieve and player meshes.
    intersection_V, _ = mesh_boolean(mesh_V,
                                    mesh_F,
                                    sieve_V_transformed,
                                    sieve_F,
                                    boolean_type='intersection')

    if len(intersection_V) > 0:
      intersection_z_low = np.min(intersection_V, axis=0)[2]
      tol = 0.5
      mesh_z_offset = (sieve_z_max + tol) - intersection_z_low
    else:
      # Mesh fits inside sieve.
      tol = 5.
      mesh_z_offset = (sieve_z_max + tol) - mesh_z_max

  return mesh_z_offset


def compute_offsets_for_meshes_in_sieve(result_name: str,
                                        sim_scale: float,
                                        mesh_configs: list[dict],
                                        mesh_rot_vecs: list[list[float]],
                                        mesh_trans_vecs: list[list[float]],
                                        mesh_blocked: list[bool]):
  '''
  Automatically compute z-axis offsets for rendering meshes with a sieve.
  '''

  result_dir = os.path.abspath(os.path.join(RESULTS_DIR, result_name))
  sieve_filepath = os.path.abspath(os.path.join(result_dir, 'sieve.stl'))
  sieve_V, sieve_F = gp.read_mesh(sieve_filepath)

  offsets = [0. for _ in mesh_configs]

  for idx, mesh_config in enumerate(mesh_configs):
    mesh_filename = mesh_config['filename']
    mesh_filepath = os.path.abspath(os.path.join(DATA_DIR, mesh_filename))
    mesh_fab_scale = mesh_config['fabrication_scale']
    mesh = mesh_utils.import_mesh(mesh_filepath, mesh_fab_scale)

    inv_sim_scale = 1. / sim_scale
    mesh_transformed = mesh_utils.get_transformed_mesh(mesh,
                                                       torch.tensor(mesh_rot_vecs[idx]).cuda(),
                                                       inv_sim_scale * torch.tensor(mesh_trans_vecs[idx]).cuda())
    mesh_transformed_V = mesh_transformed.vertices.detach().clone().cpu().numpy()
    mesh_transformed_F = mesh_transformed.faces.detach().clone().cpu().numpy()

    offsets[idx] = compute_offset_for_mesh_in_sieve(mesh_transformed_V, mesh_transformed_F, sieve_V, sieve_F, mesh_blocked[idx])

  return offsets


def render_mesh_in_sieve(result_name: str,
                         sim_scale: float,
                         mesh_filename: str,
                         mesh_color: bt.colorObj,
                         mesh_fab_scale: float,
                         mesh_rot_vec: tuple[float, float, float, float, float, float],
                         mesh_trans_vec: tuple[float, float] = (0., 0.),
                         mesh_pos_z: float = 0.6,
                         mesh_use_edge_splitting: bool = False,
                         mesh_use_smooth_shading: bool = True,
                         sieve_color: bt.colorObj = None,
                         sieve_pos_z: float = 0.3,
                         pos_xy: tuple[float, float] = (0.2, 0.0),
                         rot: tuple[float, float, float] = (0, 0, 45),
                         scale: float = 0.01,
                         shadow_brightness: float = None,
                         light_sun_angle: tuple[int, int, int] = (6, -30, -155),
                         light_sun_strength: int = 4,
                         light_sun_shadow_softness: float = 0.3,
                         light_ambient_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0),
                         cam_location: tuple[float, float, float] = (1., 0., 3.),
                         cam_rotation: tuple[float, float, float] = (15, 0, 90),
                         focal_length: int = 45,
                         is_A: bool = None,
                         save_filepath: str = None):
  '''
  Render a player mesh with a sieve.
  '''

  # Both meshes share the same scale, rotation, and XY-plane positions.
  scale_xyz = (scale, scale, scale)

  # Set up mesh.
  mesh_filepath = os.path.abspath(os.path.join(DATA_DIR, mesh_filename))
  mesh = mesh_utils.import_mesh(mesh_filepath, mesh_fab_scale)
  inv_sim_scale = 1. / sim_scale
  mesh_transformed = mesh_utils.get_transformed_mesh(mesh,
                                                     torch.tensor(mesh_rot_vec).cuda(),
                                                     inv_sim_scale * torch.tensor(mesh_trans_vec).cuda())
  mesh_transformed_V = mesh_transformed.vertices.detach().clone().cpu().numpy()
  mesh_transformed_F = mesh_transformed.faces.detach().clone().cpu().numpy()
  mesh_pos = (*pos_xy, mesh_pos_z)
  mesh = bt.readNumpyMesh(mesh_transformed_V, mesh_transformed_F, mesh_pos, rot, scale_xyz)
  bt.setMat_plastic(mesh, mesh_color)

  # Set up sieve mesh.
  result_dir = os.path.abspath(os.path.join(RESULTS_DIR, result_name))
  sieve_filepath = os.path.abspath(os.path.join(result_dir, 'sieve.stl'))
  sieve_V, sieve_F = gp.read_mesh(sieve_filepath)
  sieve_V_smoothed, sieve_F_smoothed = smooth_mesh(sieve_V, sieve_F)
  sieve_pos = (*pos_xy, sieve_pos_z)
  sieve_mesh = bt.readNumpyMesh(sieve_V_smoothed, sieve_F_smoothed, sieve_pos, rot, scale_xyz)
  bt.setMat_plastic(sieve_mesh, sieve_color)

  # Set camera (ideally change mesh instead of camera, unless you want to adjust the elevation).
  cam = bt.setCamera_from_UI(cam_location, cam_rotation, focal_length)

  # Get Blender mesh objects.
  mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

  # Split edges for mesh.
  if mesh_use_edge_splitting:
    split_edges(mesh_objects[0])

  # Set shading for mesh.
  if mesh_use_smooth_shading:
    with bpy.context.temp_override(selected_editable_objects=[mesh_objects[0]]):
      bpy.ops.object.shade_smooth()

  # Set shading for sieve mesh.
  with bpy.context.temp_override(selected_editable_objects=[mesh_objects[1]]):
    bpy.ops.object.shade_smooth() 

  # Split edges for sieve mesh.
  split_edges(mesh_objects[1])

  # Set invisible plane (shadow catcher).
  if shadow_brightness is not None:
    bt.invisibleGround(shadowBrightness=shadow_brightness)

  # Set sun light.
  if light_sun_strength is not None:
    _sun = bt.setLight_sun(light_sun_angle, light_sun_strength, light_sun_shadow_softness)

  # Set ambient light.
  if light_ambient_color is not None:
    bt.setLight_ambient(color=light_ambient_color) 

  # Set gray shadow to completely white with a threshold.
  bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

  # Save output files.
  mesh_name = mesh_filename.rsplit('.', 1)[0]

  if save_filepath is None:
    if is_A is not None:
      player = 'A' if is_A else 'B'
      blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{player}_{mesh_name}_in_sieve.blend'))
      img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{player}_{mesh_name}_in_sieve.png'))
    else:
      blend_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{mesh_name}_in_sieve.blend'))
      img_filepath = os.path.abspath(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_{mesh_name}_in_sieve.png'))
  else:
    blend_filepath = f'{save_filepath}.blend'
    img_filepath = f'{save_filepath}.png'

  bpy.ops.wm.save_mainfile(filepath=blend_filepath)
  bt.renderImage(img_filepath, cam)

  return


def render_meshes_in_sieve(result_name: str,
                           sim_scale: float,
                           mesh_color: bt.colorObj,
                           mesh_configs: list[dict],
                           mesh_rot_vecs: list[list[float]],
                           mesh_trans_vecs: list[list[float]],
                           mesh_losses: list[float],
                           z_offsets: list[float],
                           default_z_offset: float,
                           mesh_use_edge_splitting_flags: list[bool],
                           mesh_use_smooth_shading_flags: list[bool],
                           res: tuple[int, int],
                           num_samples: int,
                           is_A: bool,
                           sieve_pos_z: float = 0.3,
                           pos_xy: tuple[float, float] = (0.2, 0.0),
                           rot: tuple[float, float, float] = (0, 0, 45),
                           scale: float = 0.01,
                           shadow_brightness: float = None,
                           light_sun_angle: tuple[int, int, int] = (6, -30, -155),
                           light_sun_strength: int = 4,
                           light_sun_shadow_softness: float = 0.3,
                           light_ambient_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0),
                           cam_location: tuple[float, float, float] = (1., 0., 3.),
                           cam_rotation: tuple[float, float, float] = (15, 0, 90),
                           focal_length: int = 45):
  '''
  Create a render of each mesh being in or above the sieve.
  '''

  if mesh_use_edge_splitting_flags is None:
    mesh_use_edge_splitting_flags = [False for _ in mesh_configs]

  if mesh_use_smooth_shading_flags is None:
    mesh_use_smooth_shading_flags = [True for _ in mesh_configs]

  if z_offsets is None:
    if default_z_offset is None:
      # If no z-axis offset values are given and there is also no default value, compute them automatically.
      mesh_blocked = [loss > 0. for loss in mesh_losses]

      z_offsets_without_sieve_offset = compute_offsets_for_meshes_in_sieve(result_name,
                                                                           sim_scale,
                                                                           mesh_configs,
                                                                           mesh_rot_vecs,
                                                                           mesh_trans_vecs,
                                                                           mesh_blocked)
      z_offsets = [offset * scale + sieve_pos_z for offset in z_offsets_without_sieve_offset]
    else:
      # If no z-axis offset values are given but there is a default value, use that for all offsets.
      z_offsets = [default_z_offset for _ in mesh_configs]

  for idx, mesh_config in enumerate(mesh_configs):
    init_blender(res, num_samples)
    render_mesh_in_sieve(result_name=result_name,
                         sim_scale=sim_scale,
                         mesh_filename=mesh_config['filename'],
                         mesh_color=mesh_color,
                         mesh_fab_scale=mesh_config['fabrication_scale'],
                         mesh_rot_vec=mesh_rot_vecs[idx],
                         mesh_trans_vec=mesh_trans_vecs[idx],
                         mesh_pos_z=z_offsets[idx],
                         mesh_use_edge_splitting=mesh_use_edge_splitting_flags[idx],
                         mesh_use_smooth_shading=mesh_use_smooth_shading_flags[idx],
                         sieve_color=sieve_color,
                         is_A=is_A,
                         sieve_pos_z=sieve_pos_z,
                         pos_xy=pos_xy,
                         rot=rot,
                         scale=scale,
                         shadow_brightness=shadow_brightness,
                         light_sun_angle=light_sun_angle,
                         light_sun_strength=light_sun_strength,
                         light_sun_shadow_softness=light_sun_shadow_softness,
                         light_ambient_color=light_ambient_color,
                         cam_location=cam_location,
                         cam_rotation=cam_rotation,
                         focal_length=focal_length)

  return


def render_A_and_B_meshes_orthographic_views(result_name: str,
                                             config_dict: dict,
                                             result_dict: dict,
                                             res: tuple[int, int],
                                             num_samples: int,
                                             meshes_A_use_edge_splitting_flags: list[bool] = None,
                                             meshes_A_use_smooth_shading_flags: list[bool] = None,
                                             meshes_B_use_edge_splitting_flags: list[bool] = None,
                                             meshes_B_use_smooth_shading_flags: list[bool] = None):
  '''
  Render optimized orientations of all A and B meshes with an orthographic view.
  '''

  # Render all A meshes in orthographic view.
  for idx, mesh_A_config in enumerate(config_dict['meshes_A']):
    use_edge_splitting = False if meshes_A_use_edge_splitting_flags is None else meshes_A_use_edge_splitting_flags[idx]
    use_smooth_shading = True if meshes_A_use_smooth_shading_flags is None else meshes_A_use_smooth_shading_flags[idx]

    init_blender(res, num_samples)
    render_mesh_orthographic(result_name=result_name,
                             mesh_filename=mesh_A_config['filename'],
                             mesh_color=mesh_color_A,
                             mesh_rot_vec=result_dict['meshes_A_rot_vecs'][idx],
                             mesh_fab_scale=mesh_A_config['fabrication_scale'],
                             use_edge_splitting=use_edge_splitting,
                             use_smooth_shading=use_smooth_shading,
                             is_A=True)

  # Render all B meshes in orthographic view.
  for idx, mesh_B_config in enumerate(config_dict['meshes_B']):
    use_edge_splitting = False if meshes_B_use_edge_splitting_flags is None else meshes_B_use_edge_splitting_flags[idx]
    use_smooth_shading = True if meshes_B_use_smooth_shading_flags is None else meshes_B_use_smooth_shading_flags[idx]

    init_blender(res, num_samples)
    render_mesh_orthographic(result_name=result_name,
                             mesh_filename=mesh_B_config['filename'],
                             mesh_color=mesh_color_B,
                             mesh_rot_vec=result_dict['meshes_B_rot_vecs'][idx],
                             mesh_fab_scale=mesh_B_config['fabrication_scale'],
                             use_edge_splitting=use_edge_splitting,
                             use_smooth_shading=use_smooth_shading,
                             is_A=False)

  return


def render_rasters(result_name: str,
                   config_dict: dict,
                   result_dict: dict):
  '''
  Render figure rasters.
  '''

  simulation_scale = config_dict['simulation_scale']

  # Create the union mesh.
  meshes_A = [mesh_utils.import_mesh_from_config(mesh_A_config, DATA_DIR, simulation_scale)
              for mesh_A_config in config_dict['meshes_A']]
  meshes_A_rot_vecs = torch.tensor(result_dict['meshes_A_rot_vecs']).cuda()
  meshes_A_trans_vecs = torch.tensor(result_dict['meshes_A_trans_vecs']).cuda()
  meshes_A_transformed = [mesh_utils.get_transformed_mesh(mesh_A, meshes_A_rot_vecs[i], meshes_A_trans_vecs[i])
                          for i, mesh_A in enumerate(meshes_A)]
  union_mesh = mesh_utils.get_union_mesh(meshes_A_transformed)

  # Create a raster with each A mesh.
  for idx, mesh_A_config in enumerate(config_dict['meshes_A']):
    mesh_A = mesh_utils.import_mesh(filepath=os.path.join(DATA_DIR, mesh_A_config['filename']),
                                    fabrication_scale=mesh_A_config['fabrication_scale'],
                                    fabrication_offset=None,
                                    simulation_normalize=False,
                                    simulation_scale=simulation_scale)
    mesh_A_transformed = mesh_utils.get_transformed_mesh(mesh_A,
                                                         torch.tensor(result_dict['meshes_A_rot_vecs'][idx]).cuda(),
                                                         torch.tensor(result_dict['meshes_A_trans_vecs'][idx]).cuda())

    figure_raster = vis.get_figure_raster(union_mesh, mesh_A_transformed, torch.IntTensor(color_A).cuda())

    mesh_A_filename = mesh_A_config['filename']
    mesh_A_name = mesh_A_filename.rsplit('.', 1)[0]

    plt.figure(figsize=(2.048, 2.048))
    plt.axis('off')
    plt.imshow(figure_raster)
    plt.savefig(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_A_{mesh_A_name}_raster.png'),
                transparent=True,
                dpi=1000)

  # Create a raster with each B mesh.
  for idx, mesh_B_config in enumerate(config_dict['meshes_B']):
    mesh_B = mesh_utils.import_mesh_from_config(mesh_B_config, DATA_DIR, simulation_scale)
    mesh_B_transformed = mesh_utils.get_transformed_mesh(mesh_B,
                                                         torch.tensor(result_dict['meshes_B_rot_vecs'][idx]).cuda(),
                                                         torch.tensor(result_dict['meshes_B_trans_vecs'][idx]).cuda())

    figure_raster = vis.get_figure_raster(union_mesh, mesh_B_transformed, torch.IntTensor(color_B).cuda())

    mesh_B_filename = mesh_B_config['filename']
    mesh_B_name = mesh_B_filename.rsplit('.', 1)[0]

    plt.figure(figsize=(2.048, 2.048))
    plt.axis('off')
    plt.imshow(figure_raster)
    plt.savefig(os.path.join(FILE_DIR, result_name, f'{result_name}_mesh_B_{mesh_B_name}_raster.png'),
                transparent=True,
                dpi=1000)

  return
