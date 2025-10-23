import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
import numpy as np
import pandas as pd

import render
from sieves import mesh_utils


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('result_names_filepath', help='filepath to text file containing list of result names', type=str)
  args = parser.parse_args()

  return args


def get_meshes_data(mesh_filenames: list[str]):
  '''
  Get a dataframe of mesh info.
  '''

  num_meshes = len(mesh_filenames)
  vertex_counts = np.zeros(num_meshes, dtype=int)
  face_counts = np.zeros(num_meshes, dtype=int)

  for idx, mesh_filename in enumerate(mesh_filenames):
    mesh_filepath = os.path.abspath(os.path.join(DATA_DIR, mesh_filename))
    mesh = mesh_utils.import_mesh(mesh_filepath)

    vertex_counts[idx] = len(mesh.vertices)
    face_counts[idx] = len(mesh.faces)
  
  mesh_df = pd.DataFrame(
    {
      'mesh_filenames': mesh_filenames,
      'vertex_counts': vertex_counts,
      'face_counts': face_counts
    }
  )

  return mesh_df


def main(args):
  result_names_filepath = args.result_names_filepath

  mesh_filenames_set = set()

  meshes_A_lists = []
  meshes_B_lists = []
  optimization_times = []
  winner_declaration_times = []
  final_losses_per_B_mesh = []
  A_wins = []

  with open(result_names_filepath) as file:
    result_names = [line.rstrip() for line in file]

  for result_name in result_names:
    config_dict, result_dict = render.process_config_and_result_dicts(result_name)

    meshes_A_filenames = [mesh_config['filename'] for mesh_config in config_dict['meshes_A']]
    meshes_B_filenames = [mesh_config['filename'] for mesh_config in config_dict['meshes_B']]

    mesh_filenames_set.update(meshes_A_filenames)
    mesh_filenames_set.update(meshes_B_filenames)

    optimization_time = result_dict['total_time']
    winner_declaration_time = result_dict['winner_declaration_time']
    final_loss_per_B_mesh = result_dict['meshes_B_best_losses']

    meshes_A_lists.append(', '.join(meshes_A_filenames))
    meshes_B_lists.append(', '.join(meshes_B_filenames))
    optimization_times.append(f'{optimization_time:.1f}')
    winner_declaration_times.append(f'{winner_declaration_time:.1f}')
    final_losses_per_B_mesh.append(', '.join([f'{loss:.3f}' for loss in final_loss_per_B_mesh]))
    A_wins.append('Y' if min(final_loss_per_B_mesh) > 0 else 'N')

  dir_name = 'tables'
  figures_dir = os.path.abspath(os.path.join(render.FILE_DIR, dir_name))
  os.makedirs(figures_dir, exist_ok=True)

  # Create table of result info.
  result_df = pd.DataFrame(
    {
      'Result': result_names,
      'A meshes': meshes_A_lists,
      'B meshes': meshes_B_lists,
      'Optimization time': optimization_times,
      'Winner declaration time': winner_declaration_times,
      'Final losses per B mesh': final_losses_per_B_mesh,
      'A_wins': A_wins
    }
  )
  result_df.to_latex(index=False, buf=os.path.join(FILE_DIR, dir_name, 'table_result_data.tex'))

  # Create table of mesh info.
  mesh_filenames = list(mesh_filenames_set)
  mesh_filenames.sort()
  mesh_df = get_meshes_data(mesh_filenames)
  mesh_df.to_latex(index=False, escape=True, buf=os.path.join(FILE_DIR, dir_name, 'table_mesh_data.tex'))

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
