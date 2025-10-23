import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
import blendertoolbox as bt
import bpy
import copy
import json
from kaolin.rep import SurfaceMesh
from matplotlib import pyplot as plt
import numpy as np
import torch

import render
from sieves import batched_opt as bopt
from sieves import fabrication as fab
from sieves import mesh_utils
from sieves import orientation as orie
from sieves import particle_swarm_opt as pso
from sieves import testing
from sieves import visualization as vis
from sieves.colors import colors


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('config_path', help='filepath to input configuration JSON file', type=str)
  parser.add_argument('--opt', default=False, action='store_true')
  parser.add_argument('--render', default=False, action='store_true')
  args = parser.parse_args()

  return args


def optimize_posn_of_A_pso_objective(rot_vec_per_A_posn: torch.Tensor,
                                     mesh_A: SurfaceMesh,
                                     mesh_B: SurfaceMesh,
                                     rot_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                     trans_vec_per_B_posn_per_A_posn_per_B_mesh: torch.Tensor,
                                     opt_rot_vec: torch.Tensor,
                                     opt_trans_vec: torch.Tensor,
                                     mesh_A_fill_soft_mask_holes: bool = True,
                                     mesh_B_opt_use_area_proportion: bool = True,
                                     mesh_B_opt_vtx_out_of_bounds_loss_weight: float = 1000.,
                                     mesh_B_opt_iters: int = 100,
                                     mesh_B_opt_learning_rate: float = 0.05,
                                     image_shape: tuple[int, int] = (256, 256)):
  '''
  Slightly modified version of original function in batched_opt.py so that it returns
  the optimized positions of the B meshes via function arguments.
  '''

  meshes_B = [mesh_B]

  _, opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh, opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh = bopt.optimize_B_posn_per_A_posn_per_B_mesh(rot_vec_per_A_posn=rot_vec_per_A_posn,
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
  hard_mask_per_A_posn = bopt.get_rasters(vtxs_A_transformed_per_A_posn, mesh_A.faces, image_shape, mesh_A_fill_soft_mask_holes)
  loss_per_B_posn_per_A_posn_per_B_mesh = bopt.get_loss_per_B_posn_per_A_posn_per_B_mesh(hard_mask_per_A_posn,
                                                                                         meshes_B,
                                                                                         opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                         opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                         mesh_B_opt_use_area_proportion,
                                                                                         True,
                                                                                         mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                                                         image_shape)

  loss_per_B_posn_per_A_posn = loss_per_B_posn_per_A_posn_per_B_mesh[0]
  loss_per_A_posn, loss_per_A_posn_min_idxs = loss_per_B_posn_per_A_posn.min(dim=-1)
  max_loss_per_A_posn_idx = loss_per_A_posn.argmax()

  opt_rot_vec[:] = opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh[0, max_loss_per_A_posn_idx, loss_per_A_posn_min_idxs[max_loss_per_A_posn_idx]].detach().clone()
  opt_trans_vec[:] = opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh[0, max_loss_per_A_posn_idx, loss_per_A_posn_min_idxs[max_loss_per_A_posn_idx]].detach().clone()

  return loss_per_A_posn.detach().clone()


def get_opt_iters_values(fig_dir: str,
                         mesh_A: SurfaceMesh,
                         mesh_B: SurfaceMesh,
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
                         image_shape: tuple[int, int] = (256, 256)):

  init_A_posns = orie.generate_rotation_vectors(mesh_A_num_posns).cuda()

  rot_vec_per_B_posn = orie.generate_rotation_vectors(mesh_B_num_posns)
  trans_vec_per_B_posn = orie.generate_uniform_random_vectors(mesh_B_num_posns, 2, mesh_B_opt_posn_trans_bounds[0], mesh_B_opt_posn_trans_bounds[1])

  num_B_meshes = 1
  repeat_shape = [num_B_meshes, mesh_A_num_posns]
  rot_vec_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(rot_vec_per_B_posn.shape))])
  trans_vec_per_B_posn_per_A_posn_per_B_mesh = trans_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(trans_vec_per_B_posn.shape))])

  kwargs = {
    'mesh_A': mesh_A,
    'mesh_B': mesh_B,
    'rot_vec_per_B_posn_per_A_posn_per_B_mesh': rot_vec_per_B_posn_per_A_posn_per_B_mesh,
    'trans_vec_per_B_posn_per_A_posn_per_B_mesh': trans_vec_per_B_posn_per_A_posn_per_B_mesh,
    'opt_rot_vec': torch.zeros(6).cuda(),
    'opt_trans_vec': torch.zeros(2).cuda(),
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

  A_loss_per_iter = torch.zeros(pso_iters).cuda()
  A_rot_vec_per_iter = torch.zeros(pso_iters, 6).cuda()
  B_rot_vec_per_iter = torch.zeros(pso_iters, 6).cuda()
  B_trans_vec_per_iter = torch.zeros(pso_iters, 2).cuda()

  curr_best_val = pso_optim.best_val.detach().clone()
  curr_best_A_rot_vec = pso_optim.best_pos.detach().clone()
  curr_best_B_rot_vec = kwargs['opt_rot_vec'].detach().clone()
  curr_best_B_trans_vec = kwargs['opt_trans_vec'].detach().clone()
  
  for iter in range(pso_iters):
    pso_optim.update_velocity()
    pso_optim.update_position()
    pso_optim.evaluate_obj_func()
    pso_optim.update_best_vars()

    if pso_optim.best_val > curr_best_val:
      curr_best_val = pso_optim.best_val.detach().clone()
      curr_best_A_rot_vec = pso_optim.best_pos.detach().clone()
      curr_best_B_rot_vec = kwargs['opt_rot_vec'].detach().clone()
      curr_best_B_trans_vec = kwargs['opt_trans_vec'].detach().clone()

    A_loss_per_iter[iter] = curr_best_val
    A_rot_vec_per_iter[iter] = curr_best_A_rot_vec
    B_rot_vec_per_iter[iter] = curr_best_B_rot_vec
    B_trans_vec_per_iter[iter] = curr_best_B_trans_vec

  pso_opt_data = {
    'A_loss_per_iter': A_loss_per_iter.cpu().numpy().tolist(),
    'A_rot_vec_per_iter': A_rot_vec_per_iter.cpu().numpy().tolist(),
    'B_rot_vec_per_iter': B_rot_vec_per_iter.cpu().numpy().tolist(),
    'B_trans_vec_per_iter': B_trans_vec_per_iter.cpu().numpy().tolist()
  }

  json_filepath = os.path.join(fig_dir, 'render_opt_iters_vis_data.json')
  with open(json_filepath, 'w') as outfile:
    json.dump(pso_opt_data, outfile, indent=2)

  return


def create_loss_per_iter_plot(loss_per_iter: list[float],
                              result_name: str,
                              fig_dir: str):
  iters = np.arange(1, 1 + len(loss_per_iter))

  plt.figure(figsize=(6, 4))
  plt.plot(iters, np.array(loss_per_iter))
  plt.xlabel("PSO iteration")
  plt.ylabel("Loss")
  plt.title("Loss per PSO iteration")
  plt.grid(True)
  plt.savefig(os.path.join(fig_dir, f'{result_name}_PSO_loss_plot.svg'),
              dpi=1000)

  return


def create_rasters(result_name: str,
                   mesh_A_dilated: SurfaceMesh,
                   mesh_A: SurfaceMesh,
                   mesh_B: SurfaceMesh,
                   pso_opt_data: dict,
                   fig_dir: str):
  
  A_rot_vec_per_iter = pso_opt_data['A_rot_vec_per_iter']
  B_rot_vec_per_iter = pso_opt_data['B_rot_vec_per_iter']
  B_trans_vec_per_iter = pso_opt_data['B_trans_vec_per_iter']

  num_iters = len(A_rot_vec_per_iter)

  for iter in range(num_iters):
    A_rot_vec = torch.tensor(A_rot_vec_per_iter[iter]).cuda()
    mesh_A_dilated_transformed = mesh_utils.get_transformed_mesh(mesh_A_dilated, A_rot_vec)
    mesh_A_transformed = mesh_utils.get_transformed_mesh(mesh_A, A_rot_vec)

    mesh_A_figure_raster = vis.get_figure_raster(mesh_A_dilated_transformed,
                                                 mesh_A_transformed,
                                                 torch.IntTensor(colors['A']).cuda())

    plt.figure(figsize=(2.048, 2.048))
    plt.axis('off')
    plt.imshow(mesh_A_figure_raster)
    plt.savefig(os.path.join(fig_dir, f'{result_name}_mesh_A_raster_iter_{iter}.png'),
                transparent=True,
                dpi=1000)
    plt.close()

    B_rot_vec = torch.tensor(B_rot_vec_per_iter[iter]).cuda()
    B_trans_vec = torch.tensor(B_trans_vec_per_iter[iter]).cuda()
    mesh_B_transformed = mesh_utils.get_transformed_mesh(mesh_B, B_rot_vec, B_trans_vec)

    mesh_B_figure_raster = vis.get_figure_raster(mesh_A_dilated_transformed,
                                                 mesh_B_transformed,
                                                 torch.IntTensor(colors['B']).cuda())

    plt.figure(figsize=(2.048, 2.048))
    plt.axis('off')
    plt.imshow(mesh_B_figure_raster)
    plt.savefig(os.path.join(fig_dir, f'{result_name}_mesh_B_raster_iter_{iter}.png'),
                transparent=True,
                dpi=1000)
    plt.close()

  return


def render_mesh_in_sieve(figure_name: str,
                         fig_dir: str,
                         mesh_V: np.ndarray,
                         mesh_F: np.ndarray,
                         sieve_V: np.ndarray,
                         sieve_F: np.ndarray,
                         mesh_color: bt.colorObj,
                         mesh_pos_z: float = 0.6,
                         mesh_use_edge_splitting: bool = False,
                         mesh_use_smooth_shading: bool = True,
                         sieve_color: bt.colorObj = render.sieve_color,
                         sieve_pos_z: float = 0.3,
                         pos_xy: tuple[float, float] = (0.2, 0.0),
                         rot: tuple[float, float, float] = (0, 0, 45),
                         scale: float = 0.01,
                         light_sun_angle: tuple[int, int, int] = (6, -30, -155),
                         light_sun_strength: int = 4,
                         light_sun_shadow_softness: float = 0.3,
                         light_ambient_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0),
                         cam_location: tuple[float, float, float] = (1., 0., 3.),
                         cam_rotation: tuple[float, float, float] = (15, 0, 90),
                         focal_length: int = 45,
                         res: tuple[int, int] = (2048, 2048),
                         num_samples: int = 200):

  render.init_blender(res, num_samples)

  # All meshes share same scale.
  scale_xyz = (scale, scale, scale)

  # Set up mesh.
  mesh_pos = (*pos_xy, mesh_pos_z)
  mesh = bt.readNumpyMesh(mesh_V, mesh_F, mesh_pos, rot, scale_xyz)
  bt.setMat_plastic(mesh, mesh_color)

  # Set up sieve mesh.
  sieve_V_smoothed, sieve_F_smoothed = render.smooth_mesh(sieve_V, sieve_F)
  sieve_pos = (*pos_xy, sieve_pos_z)
  sieve_mesh = bt.readNumpyMesh(sieve_V_smoothed, sieve_F_smoothed, sieve_pos, rot, scale_xyz)
  bt.setMat_plastic(sieve_mesh, sieve_color)

  # Set camera (ideally change mesh instead of camera, unless you want to adjust the elevation).
  cam = bt.setCamera_from_UI(cam_location, cam_rotation, focal_length)

  # Get Blender mesh objects.
  mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

  # Split edges for mesh.
  if mesh_use_edge_splitting:
    render.split_edges(mesh_objects[0])

  # Set shading for mesh.
  if mesh_use_smooth_shading:
    with bpy.context.temp_override(selected_editable_objects=[mesh_objects[0]]):
      bpy.ops.object.shade_smooth()

  # Set shading for sieve mesh.
  with bpy.context.temp_override(selected_editable_objects=[mesh_objects[1]]):
    bpy.ops.object.shade_smooth() 

  # Split edges for sieve mesh.
  render.split_edges(mesh_objects[1])

  # Set sun light.
  if light_sun_strength is not None:
    _sun = bt.setLight_sun(light_sun_angle, light_sun_strength, light_sun_shadow_softness)

  # Set ambient light.
  if light_ambient_color is not None:
    bt.setLight_ambient(color=light_ambient_color) 

  # Set gray shadow to completely white with a threshold.
  bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

  # Save output files.
  blend_filepath = os.path.join(fig_dir, f'{figure_name}.blend')
  img_filepath = os.path.join(fig_dir, f'{figure_name}.png')

  bpy.ops.wm.save_mainfile(filepath=blend_filepath)
  bt.renderImage(img_filepath, cam)

  return


def create_renders_of_meshes_in_sieve(result_name: str,
                                      simulation_scale: float,
                                      mesh_A_dilated: SurfaceMesh,
                                      mesh_A: SurfaceMesh,
                                      mesh_B: SurfaceMesh,
                                      pso_opt_data: dict,
                                      fig_dir: str):
  sieve_pos_z = 0.3

  A_rot_vec_per_iter = pso_opt_data['A_rot_vec_per_iter']
  B_rot_vec_per_iter = pso_opt_data['B_rot_vec_per_iter']
  B_trans_vec_per_iter = pso_opt_data['B_trans_vec_per_iter']

  num_iters = len(A_rot_vec_per_iter)

  for iter in range(num_iters):
    A_rot_vec = torch.tensor(A_rot_vec_per_iter[iter]).cuda()
    mesh_A_dilated_transformed = mesh_utils.get_transformed_mesh(mesh_A_dilated, A_rot_vec)

    mesh_scale_factor = 1. / simulation_scale
    sieve_V, sieve_F = fab.get_sieve_plate_minus_mesh(mesh=mesh_A_dilated_transformed,
                                                      scale_factor=mesh_scale_factor,
                                                      fill_holes=True,
                                                      sieve_side_length=80,
                                                      sieve_thickness=60,
                                                      image_shape=(1024, 1024))

    # Render mesh A in sieve.
    mesh_A_scaled = copy.deepcopy(mesh_A)
    mesh_A_scaled.vertices *= mesh_scale_factor
    mesh_A_transformed = mesh_utils.get_transformed_mesh(mesh_A_scaled, A_rot_vec)
    mesh_A_transformed_V = mesh_A_transformed.vertices.detach().clone().cpu().numpy()
    mesh_A_transformed_F = mesh_A_transformed.faces.detach().clone().cpu().numpy()

    render_mesh_in_sieve(figure_name=f'{result_name}_mesh_A_in_sieve_iter_{iter}',
                         fig_dir=fig_dir,
                         mesh_V=mesh_A_transformed_V,
                         mesh_F=mesh_A_transformed_F,
                         sieve_V=sieve_V,
                         sieve_F=sieve_F,
                         mesh_color=render.mesh_color_A,
                         mesh_pos_z=0.5,
                         sieve_pos_z=sieve_pos_z,
                         scale=simulation_scale,
                         cam_location=(2., 0., 4.),
                         cam_rotation=(25, 0, 90))

    # Render mesh B in sieve.
    B_rot_vec = torch.tensor(B_rot_vec_per_iter[iter]).cuda()
    B_trans_vec = torch.tensor(B_trans_vec_per_iter[iter]).cuda()
    mesh_B_scaled = copy.deepcopy(mesh_B)
    mesh_B_scaled.vertices *= mesh_scale_factor
    mesh_B_transformed = mesh_utils.get_transformed_mesh(mesh_B_scaled, B_rot_vec, mesh_scale_factor * B_trans_vec)
    mesh_B_transformed_V = mesh_B_transformed.vertices.detach().clone().cpu().numpy()
    mesh_B_transformed_F = mesh_B_transformed.faces.detach().clone().cpu().numpy()

    mesh_offset_without_sieve_offset = render.compute_offset_for_mesh_in_sieve(mesh_B_transformed_V,
                                                                               mesh_B_transformed_F,
                                                                               sieve_V,
                                                                               sieve_F,
                                                                               True)
    mesh_pos_z = mesh_offset_without_sieve_offset * simulation_scale + sieve_pos_z

    render_mesh_in_sieve(figure_name=f'{result_name}_mesh_B_in_sieve_iter_{iter}',
                         fig_dir=fig_dir,
                         mesh_V=mesh_B_transformed_V,
                         mesh_F=mesh_B_transformed_F,
                         sieve_V=sieve_V,
                         sieve_F=sieve_F,
                         mesh_color=render.mesh_color_B,
                         mesh_pos_z=mesh_pos_z,
                         sieve_pos_z=sieve_pos_z,
                         scale=simulation_scale,
                         cam_location=(2., 0., 4.),
                         cam_rotation=(25, 0, 90))

  return


def main(args):
  config_filepath = args.config_path
  run_all = not (args.opt or args.render)

  fig_dir = os.path.join(FILE_DIR, 'opt_iters_vis')
  os.makedirs(fig_dir, exist_ok=True)

  '''
  Read the configuration file and import meshes.
  '''

  with open(config_filepath, 'r') as config_file:
    config_dict = json.load(config_file) 

  result_name = config_dict['name']
  simulation_scale = config_dict['simulation_scale']

  mesh_A_config = config_dict['meshes_A'][0]
  mesh_A = mesh_utils.import_mesh(filepath=os.path.abspath(os.path.join(DATA_DIR, mesh_A_config['filename'])),
                                  fabrication_scale=mesh_A_config['fabrication_scale'],
                                  fabrication_offset=None,
                                  simulation_normalize=False,
                                  simulation_scale=simulation_scale)

  mesh_B_config = config_dict['meshes_B'][0]
  mesh_B = mesh_utils.import_mesh(filepath=os.path.abspath(os.path.join(DATA_DIR, mesh_B_config['filename'])),
                                  fabrication_scale=mesh_B_config['fabrication_scale'],
                                  fabrication_offset=None,
                                  simulation_normalize=False,
                                  simulation_scale=simulation_scale)

  '''
  Get PSO optimization data per iteration.
  '''

  if run_all or args.opt:
    mesh_A_dilated = mesh_utils.import_mesh(filepath=os.path.abspath(os.path.join(DATA_DIR, mesh_A_config['filename'])),
                                            fabrication_scale=mesh_A_config['fabrication_scale'],
                                            fabrication_offset=mesh_A_config['fabrication_offset'],
                                            simulation_normalize=False,
                                            simulation_scale=simulation_scale)

    get_opt_iters_values(fig_dir, mesh_A_dilated, mesh_B)

    print('Computed PSO optimization data.')

  '''
  Create all renders.
  '''

  if run_all or args.render:
    json_filepath = os.path.join(fig_dir, 'render_opt_iters_vis_data.json')
    with open(json_filepath, 'r') as file:
      pso_opt_data = json.load(file)

    # Create loss plot.
    create_loss_per_iter_plot(pso_opt_data['A_loss_per_iter'], result_name, fig_dir)
    print('Created loss plot.')

    # Create rasters.
    create_rasters(result_name, mesh_A_dilated, mesh_A, mesh_B, pso_opt_data, fig_dir)
    print('Created rasters.')

    # Create renders of meshes in sieve.
    create_renders_of_meshes_in_sieve(result_name, simulation_scale, mesh_A_dilated, mesh_A, mesh_B, pso_opt_data, fig_dir)
    print('Created renders of meshes in sieve.')

  return


if __name__ == '__main__':
  testing.set_determinism()
  args = parse_args()
  main(args)
