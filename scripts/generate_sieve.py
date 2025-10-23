import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
from datetime import timedelta
import json
from kaolin.rep import SurfaceMesh
from matplotlib import pyplot as plt
import time
import torch

from sieves import mesh_utils
from sieves import batched_opt as bopt
from sieves import fabrication as fab
from sieves import testing
from sieves import visualization as vis


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('config_path', help='filepath to input configuration JSON file', type=str)
  parser.add_argument('results_path', help='filepath to save results', type=str)
  parser.add_argument('--opt', default=False, action='store_true')
  parser.add_argument('--sieve', default=False, action='store_true')
  parser.add_argument('--verify', default=False, action='store_true')
  args = parser.parse_args()

  return args


def generate_sieve_mesh(config: dict,
                        union_mesh: SurfaceMesh,
                        RESULTS_DIR: str):
  '''
  Create and save sieve mesh.
  '''

  mesh_scale_factor = 1. / config['simulation_scale']
  sieve_mesh_V, sieve_mesh_F = fab.get_sieve_plate_minus_mesh(mesh=union_mesh,
                                                              scale_factor=mesh_scale_factor,
                                                              fill_holes=config['mesh_A_fill_soft_mask_holes'],
                                                              sieve_side_length=config['sieve_side_length'],
                                                              sieve_thickness=config['sieve_thickness'],
                                                              image_shape=config['sieve_resolution'])

  mesh_utils.export_mesh_as_stl(os.path.join(RESULTS_DIR, 'sieve.stl'), sieve_mesh_V, sieve_mesh_F)

  return


def optimize_B_meshes(config: dict,
                      union_mesh: SurfaceMesh,
                      meshes_B: list[SurfaceMesh],
                      RESULTS_DIR: str):
  '''
  Optimize each mesh B against the union mesh and save the resulting raster.
  '''

  time_start = time.time()

  num_meshes_B = len(meshes_B)
  meshes_B_best_losses = torch.zeros(num_meshes_B)
  meshes_B_rot_vecs = torch.zeros(num_meshes_B, 6)
  meshes_B_trans_vecs = torch.zeros(num_meshes_B, 2)

  for i in range(num_meshes_B):
    mesh_B = meshes_B[i]

    best_loss, best_rot_vec, best_trans_vec = bopt.optimize_posn_of_mesh_B(union_mesh=union_mesh,
                                                                           mesh_B=mesh_B,
                                                                           union_mesh_fill_soft_mask_holes=config['mesh_A_fill_soft_mask_holes'],
                                                                           mesh_B_opt_num_posns=100,
                                                                           mesh_B_opt_posn_trans_bounds=config['mesh_B_opt_posn_trans_bounds'],
                                                                           mesh_B_opt_use_area_proportion=config['mesh_B_opt_use_area_proportion'],
                                                                           mesh_B_opt_vtx_out_of_bounds_loss_weight=config['mesh_B_opt_vtx_out_of_bounds_loss_weight'],
                                                                           mesh_B_opt_iters=500,
                                                                           mesh_B_opt_learning_rate=config['mesh_B_opt_learning_rate'],
                                                                           image_shape=config['image_shape'])

    # Record optimization result for mesh B.
    meshes_B_best_losses[i] = best_loss
    meshes_B_rot_vecs[i] = best_rot_vec
    meshes_B_trans_vecs[i] = best_trans_vec

    mesh_B_transformed = mesh_utils.get_transformed_mesh(mesh_B, best_rot_vec, best_trans_vec)
    mesh_B_raster = vis.rasterize_mesh(mesh_B_transformed, pixel_color=torch.tensor([0., 1., 0.]).cuda(), image_shape=config['image_shape'])
    union_mesh_raster = vis.rasterize_mesh(union_mesh, pixel_color=torch.tensor([1., 0., 0.]).cuda(), image_shape=config['image_shape'], fill_holes=config['mesh_A_fill_soft_mask_holes'])
    combined_raster = union_mesh_raster + mesh_B_raster

    plt.figure()
    _, plts = plt.subplots(1,1)
    plts.imshow(combined_raster.detach().clone().cpu().numpy())
    plt.savefig(os.path.join(RESULTS_DIR, f'mesh_B_{i}_opt_raster.png'))

  time_end = time.time()
  time_elapsed = time_end - time_start

  return meshes_B_best_losses, meshes_B_rot_vecs, meshes_B_trans_vecs, time_elapsed


def optimize_A_meshes(config: dict,
                      meshes_A: list[SurfaceMesh],
                      meshes_B: list[SurfaceMesh],
                      RESULTS_DIR: str):
  '''
  Optimize orientations of the A meshes.
  '''

  meshes_A_winning_rot_vecs = None if config['meshes_A_winning_rot_vecs'] is None else torch.tensor(config['meshes_A_winning_rot_vecs']).cuda()

  time_start = time.time()

  best_val, union_mesh, rot_vecs, trans_vecs = bopt.optimize_posn_of_multi_A(meshes_A=meshes_A,
                                                                             meshes_B=meshes_B,
                                                                             meshes_A_winning_rot_vecs=meshes_A_winning_rot_vecs,
                                                                             mesh_A_opt_num_posns=config['mesh_A_opt_num_posns'],
                                                                             mesh_A_fill_soft_mask_holes=config['mesh_A_fill_soft_mask_holes'],
                                                                             mesh_B_opt_num_posns=config['mesh_B_opt_num_posns'],
                                                                             mesh_B_opt_posn_trans_bounds=config['mesh_B_opt_posn_trans_bounds'],
                                                                             mesh_B_opt_use_area_proportion=config['mesh_B_opt_use_area_proportion'],
                                                                             mesh_B_opt_vtx_out_of_bounds_loss_weight=config['mesh_B_opt_vtx_out_of_bounds_loss_weight'],
                                                                             mesh_B_opt_iters=config['mesh_B_opt_iters'],
                                                                             mesh_B_opt_learning_rate=config['mesh_B_opt_learning_rate'],
                                                                             mesh_A_pso_c1=config['mesh_A_pso_c1'],
                                                                             mesh_A_pso_c2=config['mesh_A_pso_c2'],
                                                                             mesh_A_pso_w=config['mesh_A_pso_w'],
                                                                             mesh_A_pso_iters=config['mesh_A_pso_iters'],
                                                                             proj_A_pso_c1=config['proj_A_pso_c1'],
                                                                             proj_A_pso_c2=config['proj_A_pso_c2'],
                                                                             proj_A_pso_w=config['proj_A_pso_w'],
                                                                             proj_A_pso_iters=config['proj_A_pso_iters'],
                                                                             pso_verbose=config['pso_verbose'],
                                                                             image_shape=config['image_shape'])

  time_end = time.time()
  meshes_A_opt_time_elapsed = time_end - time_start

  '''
  Optimize orientations of B meshes.
  '''

  meshes_B_best_losses, meshes_B_rot_vecs, meshes_B_trans_vecs, meshes_B_opt_time_elapsed = optimize_B_meshes(config, union_mesh, meshes_B, RESULTS_DIR)

  '''
  Output and save results as JSON.
  '''

  print()
  print(f'Best loss: {best_val}')
  print(f'Best rotation vectors for A meshes: {rot_vecs}')
  print(f'Best translation vectors for A meshes: {trans_vecs}')
  print(f'Total time: {str(timedelta(seconds=meshes_A_opt_time_elapsed))}')
  print()
  print(f'Saving results to directory: {RESULTS_DIR}')

  results = {
    'total_time': meshes_A_opt_time_elapsed,
    'winner_declaration_time': meshes_B_opt_time_elapsed,
    'best_loss': best_val.item(),
    'meshes_A_rot_vecs': rot_vecs.cpu().numpy().tolist(),
    'meshes_A_trans_vecs': trans_vecs.cpu().numpy().tolist(),
    'meshes_B_best_losses': meshes_B_best_losses.cpu().numpy().tolist(),
    'meshes_B_rot_vecs': meshes_B_rot_vecs.cpu().numpy().tolist(),
    'meshes_B_trans_vecs': meshes_B_trans_vecs.cpu().numpy().tolist()
  }

  result_json_filepath = os.path.join(RESULTS_DIR, 'results.json')
  with open(result_json_filepath, 'w') as outfile:
    json.dump(results, outfile, indent=2)

  return union_mesh


def main(args):
  '''
  Read the configuration file.
  '''

  config_filepath = os.path.abspath(args.config_path)
  with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

  print(f'Config path: {config_filepath}')
  print(f'Config name: {config["name"]}')
  print()

  '''
  Load meshes.
  '''

  DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
  simulation_scale = config['simulation_scale']

  meshes_A = [mesh_utils.import_mesh_from_config(mesh_A_config, DATA_DIR, simulation_scale) for mesh_A_config in config['meshes_A']]
  meshes_B = [mesh_utils.import_mesh_from_config(mesh_B_config, DATA_DIR, simulation_scale) for mesh_B_config in config['meshes_B']]

  '''
  Run each stage of the sieve generation pipeline as necessary.
  '''

  RESULTS_DIR = os.path.abspath(args.results_path)
  os.makedirs(RESULTS_DIR, exist_ok=True)

  run_all = not (args.opt or args.sieve or args.verify)

  '''
  Optimize orientations of the A meshes.
  '''

  if run_all or args.opt:
    union_mesh = optimize_A_meshes(config, meshes_A, meshes_B, RESULTS_DIR)
  else:
    result_json_filepath = os.path.abspath(os.path.join(RESULTS_DIR, 'results.json'))
    with open(result_json_filepath, 'r') as result_json_file:
      result_dict = json.load(result_json_file)

    meshes_A_rot_vecs = torch.tensor(result_dict['meshes_A_rot_vecs']).cuda()
    meshes_A_trans_vecs = torch.tensor(result_dict['meshes_A_trans_vecs']).cuda()
    meshes_A_transformed = [mesh_utils.get_transformed_mesh(mesh_A, meshes_A_rot_vecs[i], meshes_A_trans_vecs[i])
                            for i, mesh_A in enumerate(meshes_A)]
    union_mesh = mesh_utils.get_union_mesh(meshes_A_transformed)

  '''
  Create and save sieve mesh.
  '''

  if run_all or args.sieve:
    generate_sieve_mesh(config, union_mesh, RESULTS_DIR)

  '''
  Save rasters of optimized positions of each mesh B against the union mesh.
  '''

  if args.verify and not (run_all or args.opt):
    # Optimize B meshes.
    meshes_B_best_losses, meshes_B_rot_vecs, meshes_B_trans_vecs, meshes_B_opt_time_elapsed = optimize_B_meshes(config, union_mesh, meshes_B, RESULTS_DIR)

    # Read the results file.
    result_json_filepath = os.path.abspath(os.path.join(RESULTS_DIR, 'results.json'))
    with open(result_json_filepath, 'r') as result_json_file:
      result_dict = json.load(result_json_file)

    # Update values.
    result_dict['winner_declaration_time'] = meshes_B_opt_time_elapsed
    result_dict['meshes_B_best_losses'] = meshes_B_best_losses.cpu().numpy().tolist()
    result_dict['meshes_B_rot_vecs'] = meshes_B_rot_vecs.cpu().numpy().tolist()
    result_dict['meshes_B_trans_vecs'] = meshes_B_trans_vecs.cpu().numpy().tolist()

    # Save new results.
    with open(result_json_filepath, 'w') as outfile:
      json.dump(result_dict, outfile, indent=2)

  return


if __name__ == '__main__':
  testing.set_determinism()
  args = parse_args()
  main(args)
