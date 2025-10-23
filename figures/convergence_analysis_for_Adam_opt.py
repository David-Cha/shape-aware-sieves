import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
import json
from kaolin.rep import SurfaceMesh
from matplotlib import pyplot as plt
import torch

from sieves import batched_opt as bopt
from sieves import mesh_utils
from sieves import orientation as orie
from sieves import testing


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('config_filepath', type=str)
  parser.add_argument('-n', '--mesh_B_posns_vals', type=int, nargs='+', default=[5, 10, 20, 40, 80])
  args = parser.parse_args()

  return args


def get_loss_per_iter(mesh_A: SurfaceMesh,
                      mesh_B: SurfaceMesh,
                      mesh_A_num_posns: int,
                      mesh_B_num_posns: int,
                      mesh_A_fill_soft_mask_holes: bool,
                      mesh_B_opt_posn_trans_bounds: tuple[float, float],
                      mesh_B_opt_use_area_proportion: bool,
                      mesh_B_opt_vtx_out_of_bounds_loss_weight: float,
                      mesh_B_opt_iters: int,
                      mesh_B_opt_learning_rate: float,
                      image_shape: tuple[int, int]):

  losses = torch.zeros(mesh_B_opt_iters + 1).cuda()

  init_A_posns = orie.generate_rotation_vectors(mesh_A_num_posns).cuda()

  rot_vec_per_B_posn = orie.generate_rotation_vectors(mesh_B_num_posns)
  trans_vec_per_B_posn = orie.generate_uniform_random_vectors(mesh_B_num_posns, 2, mesh_B_opt_posn_trans_bounds[0], mesh_B_opt_posn_trans_bounds[1])

  num_B_meshes = 1
  repeat_shape = [num_B_meshes, mesh_A_num_posns]
  rot_vec_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(rot_vec_per_B_posn.shape))])
  trans_vec_per_B_posn_per_A_posn_per_B_mesh = trans_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(trans_vec_per_B_posn.shape))])

  _, _, _ = bopt.optimize_B_posn_per_A_posn_per_B_mesh(rot_vec_per_A_posn=init_A_posns,
                                                       mesh_A=mesh_A,
                                                       meshes_B=[mesh_B],
                                                       rot_vec_per_B_posn_per_A_posn_per_B_mesh=rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                       trans_vec_per_B_posn_per_A_posn_per_B_mesh=trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                       mesh_A_fill_soft_mask_holes=mesh_A_fill_soft_mask_holes,
                                                       mesh_B_opt_use_area_proportion=mesh_B_opt_use_area_proportion,
                                                       mesh_B_opt_vtx_out_of_bounds_loss_weight=mesh_B_opt_vtx_out_of_bounds_loss_weight,
                                                       mesh_B_opt_iters=mesh_B_opt_iters,
                                                       mesh_B_opt_learning_rate=mesh_B_opt_learning_rate,
                                                       losses=losses,
                                                       image_shape=image_shape)

  return losses


def get_all_losses_per_mesh_B_posn_val(mesh_A: SurfaceMesh,
                                       mesh_B: SurfaceMesh,
                                       config_dict: dict,
                                       mesh_B_posns_vals: list[int],
                                       fig_dir: str):

  result_name = config_dict['name']

  for num_posns in mesh_B_posns_vals:
    losses = get_loss_per_iter(mesh_A,
                               mesh_B,
                               config_dict['mesh_A_opt_num_posns'],
                               num_posns,
                               config_dict['mesh_A_fill_soft_mask_holes'],
                               config_dict['mesh_B_opt_posn_trans_bounds'],
                               config_dict['mesh_B_opt_use_area_proportion'],
                               config_dict['mesh_B_opt_vtx_out_of_bounds_loss_weight'],
                               config_dict['mesh_B_opt_iters'],
                               config_dict['mesh_B_opt_learning_rate'],
                               config_dict['image_shape'])

    experiment_name = f'{result_name}_losses_for_{num_posns}_mesh_B_posns'

    # Save tensor.
    torch.save(losses, os.path.join(fig_dir, f'{experiment_name}_tensor.pt'))

    # Save plot.
    plt.figure(figsize=(10, 6))
    plt.plot(losses.detach().clone().cpu().numpy())
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Adam optimization loss vs Iteration for Case {result_name} with {num_posns} Mesh B Positions')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, f'{experiment_name}_plot.svg'),
                format='svg',
                transparent=True,
                dpi=500)

  return


def create_combined_plot(result_name: str,
                         mesh_B_posns_vals: list[int],
                         fig_dir: str):
  
  losses_per_mesh_B_posn_val = [None for _ in mesh_B_posns_vals]

  # Get losses.
  for idx, num_posns in enumerate(mesh_B_posns_vals):
    experiment_name = f'{result_name}_losses_for_{num_posns}_mesh_B_posns'
    tensor_filename = f'{experiment_name}_tensor.pt'
    losses = torch.load(os.path.join(fig_dir, tensor_filename))
    losses_per_mesh_B_posn_val[idx] = losses.detach().clone().cpu().numpy()

  # Create plot.
  x = range(len(losses_per_mesh_B_posn_val[0]))
  plt.figure(figsize=(10, 6))
  for posns, losses in zip(mesh_B_posns_vals, losses_per_mesh_B_posn_val):
    plt.plot(x, losses, label=f'{posns}')
  plt.legend()
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title(f'Adam optimization loss vs Iteration for Case {result_name}')
  plt.grid(True)
  plt.savefig(os.path.join(fig_dir, f'{result_name}_Adam_convergence_plot.svg'),
              format='svg',
              transparent=True,
              dpi=500)

  return


def main(args):
  config_filepath = args.config_filepath
  mesh_B_posns_vals = args.mesh_B_posns_vals

  # Create directory to store all results and figures.
  fig_dir = os.path.join(FILE_DIR, 'convergence_analysis_for_Adam_opt')
  os.makedirs(fig_dir, exist_ok=True)

  # Read config file.
  with open(config_filepath, 'r') as config_file:
    config_dict = json.load(config_file) 

  result_name = config_dict['name']
  simulation_scale = config_dict['simulation_scale']

  mesh_A_config = config_dict['meshes_A'][0]
  mesh_A = mesh_utils.import_mesh(filepath=os.path.abspath(os.path.join(DATA_DIR, mesh_A_config['filename'])),
                                  fabrication_scale=mesh_A_config['fabrication_scale'],
                                  fabrication_offset=mesh_A_config['fabrication_offset'],
                                  simulation_normalize=False,
                                  simulation_scale=simulation_scale)

  mesh_B_config = config_dict['meshes_B'][0]
  mesh_B = mesh_utils.import_mesh(filepath=os.path.abspath(os.path.join(DATA_DIR, mesh_B_config['filename'])),
                                  fabrication_scale=mesh_B_config['fabrication_scale'],
                                  fabrication_offset=None,
                                  simulation_normalize=False,
                                  simulation_scale=simulation_scale)

  get_all_losses_per_mesh_B_posn_val(mesh_A, mesh_B, config_dict, mesh_B_posns_vals, fig_dir)

  create_combined_plot(result_name, mesh_B_posns_vals, fig_dir)

  return


if __name__ == '__main__':
  testing.set_determinism()
  args = parse_args()
  main(args)
