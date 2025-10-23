import os

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'results'))
SCRIPTS_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'scripts'))

import argparse
import json
from matplotlib import pyplot as plt
import numpy as np


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('result_name', type=str)
  parser.add_argument('param', type=str)
  parser.add_argument('list_filepath', help='filepath to text file containing list of job filenames', type=str)
  args = parser.parse_args()

  return args


def process_config_and_result_dicts(result_name: str):
  '''
  Read the configuration and results JSON files, plus scale up
  the translation vectors accordingly.
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


def get_runtimes_per_param_value(param: str,
                                 result_names: list[str]):
  '''
  From the list of results, get the parameter values and the total runtimes.
  '''

  num_results = len(result_names)
  param_values = np.zeros(num_results, dtype=int)
  runtimes = np.zeros(num_results)

  for i, result_name in enumerate(result_names):
    config_dict, result_dict = process_config_and_result_dicts(result_name)

    param_values[i] = config_dict[param]
    runtimes[i] = result_dict['total_time']

  return param_values, runtimes


def create_runtime_plot(result_name: str,
                        param: str,
                        param_values: list[int],
                        runtimes: list[float]):
  '''
  Create and save plot of runtime as a function of the specified parameter.
  '''

  xlabels = {
    'mesh_A_opt_num_posns': 'Number of PSO Particles',
    'mesh_B_opt_num_posns': 'Number of Mesh B Orientations',
    'mesh_B_opt_iters': 'Adam Iterations',
    'mesh_A_pso_iters': 'PSO Iterations'
  }
  xlabel = xlabels[param]

  save_dir = os.path.abspath(os.path.join(FILE_DIR, 'runtime_analysis'))
  os.makedirs(save_dir, exist_ok=True)

  plt.figure(figsize=(8, 6))
  plt.plot(param_values, runtimes)
  plt.xscale('log', base=2)
  plt.xlabel(xlabel)
  plt.yscale('log', base=2)
  plt.ylabel('Runtime (s)')
  plt.title(f'Runtime vs {xlabel} for Case {result_name}')
  plt.grid(True)
  plt.savefig(os.path.join(save_dir, f'{result_name}_{param}_runtime_plot.svg'),
              format='svg',
              transparent=True,
              dpi=500)
  
  return


def main(args):
  result_name = args.result_name
  param = args.param
  list_filepath = args.list_filepath

  with open(list_filepath) as file:
    result_names = [line.rstrip().rsplit('.', 1)[0] for line in file]

  param_values, runtimes = get_runtimes_per_param_value(param, result_names)

  create_runtime_plot(result_name, param, param_values, runtimes)

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
