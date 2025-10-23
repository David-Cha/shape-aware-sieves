import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
from matplotlib import pyplot as plt
import numpy as np

import render
from sieves import mesh_utils


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('list_filepath', help='filepath to text file containing list of result names', type=str)
  args = parser.parse_args()

  return args


def count_total_meshes_B_vtxs(meshes_B_configs: list[dict]):
  total_vtxs = 0

  for config in meshes_B_configs:
    mesh = mesh_utils.import_mesh_from_config(config, DATA_DIR, 1.0)
    total_vtxs += len(mesh.vertices)

  return total_vtxs


def main(args):
  list_filepath = args.list_filepath

  with open(list_filepath) as file:
    result_names = [line.rstrip() for line in file]

  num_results = len(result_names)
  num_B_meshes = np.zeros(num_results)
  num_total_vtxs = np.zeros(num_results)
  opt_runtimes = np.zeros(num_results)
  winner_declaration_runtimes = np.zeros(num_results)

  for i, result_name in enumerate(result_names):
    config_dict, result_dict = render.process_config_and_result_dicts(result_name)

    meshes_B_configs = config_dict['meshes_B']
    num_B_meshes[i] = len(meshes_B_configs)
    num_total_vtxs[i] = count_total_meshes_B_vtxs(meshes_B_configs)

    opt_runtimes[i] = result_dict['total_time']
    winner_declaration_runtimes[i] = result_dict['winner_declaration_time']

  save_dir = os.path.abspath(os.path.join(FILE_DIR, 'runtime_analysis_of_mesh_count_and_complexity'))
  os.makedirs(save_dir, exist_ok=True)

  plt.figure(figsize=(8, 6))
  plt.plot(num_B_meshes, opt_runtimes, marker='o', label='optimization time')
  plt.plot(num_B_meshes, winner_declaration_runtimes, marker='o', label='winner declaration time')
  plt.legend()
  plt.xlabel('Number of B meshes')
  plt.ylabel('Runtime (s)')
  plt.title(f'Runtime vs Number of B meshes')
  plt.grid(True)
  plt.savefig(os.path.join(save_dir, f'runtime_vs_num_B_meshes_plot.svg'),
              format='svg',
              transparent=True,
              dpi=500)

  plt.figure(figsize=(8, 6))
  plt.plot(num_total_vtxs, opt_runtimes, marker='o', label='optimization time')
  plt.plot(num_total_vtxs, winner_declaration_runtimes, marker='o', label='winner declaration time')
  plt.legend()
  plt.xlabel('Number of vertices across all B meshes')
  plt.ylabel('Runtime (s)')
  plt.title(f'Runtime vs Number of vertices across all B meshes')
  plt.grid(True)
  plt.savefig(os.path.join(save_dir, f'runtime_vs_num_vertices_plot.svg'),
              format='svg',
              transparent=True,
              dpi=500)

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
