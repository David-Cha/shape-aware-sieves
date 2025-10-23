import os

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'scripts'))

import argparse
import json


default_params = {
  'meshes_A_winning_rot_vecs': None,
  'mesh_A_opt_num_posns': 10,
  'mesh_A_fill_soft_mask_holes': True,
  'mesh_B_opt_num_posns': 10,
  'mesh_B_opt_posn_trans_bounds': [-0.1, 0.1],
  'mesh_B_opt_use_area_proportion': True,
  'mesh_B_opt_vtx_out_of_bounds_loss_weight': 0,
  'mesh_B_opt_iters': 100,
  'mesh_B_opt_learning_rate': 0.05,
  'mesh_A_pso_c1': 0.25,
  'mesh_A_pso_c2': 0.25,
  'mesh_A_pso_w': 0.5,
  'mesh_A_pso_iters': 10,
  'proj_A_pso_c1': 0.05,
  'proj_A_pso_c2': 0.05,
  'proj_A_pso_w': 0.5,
  'proj_A_pso_iters': 10,
  # 'pso_verbose': True,
  'image_shape': [256, 256],
  # 'sieve_side_length': 80.0,
  # 'sieve_thickness': 60.0,
  'sieve_resolution': [1024, 1024]
}


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('result_names_filepath', help='filepath to text file containing list of result names', type=str)
  args = parser.parse_args()

  return args


def main(args):
  result_names_filepath = args.result_names_filepath

  eps = 1e-6
  results_with_non_default_params = {}

  with open(result_names_filepath) as file:
    result_names = [line.rstrip() for line in file]

  for result_name in result_names:
    curr_non_default_params = {}

    config_filepath = os.path.abspath(os.path.join(SCRIPTS_DIR, f'{result_name}.json'))
    with open(config_filepath, 'r') as config_file:
      config_dict = json.load(config_file)
    
    for param_name, default_val in default_params.items():
      is_default = True
      config_val = config_dict[param_name]

      if default_val is None:
        if config_val is not None:
          is_default = False
      elif isinstance(default_val, (bool, int)):
        if config_val != default_val:
          is_default = False
      elif isinstance(default_val, float):
        if abs(config_val - default_val) > eps:
          is_default = False

      if not is_default:
        curr_non_default_params[param_name] = config_val
    
    if len(curr_non_default_params) > 0:
      results_with_non_default_params[result_name] = curr_non_default_params

  print(json.dumps(results_with_non_default_params,
                   indent=4))

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
