import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

import argparse
from kaolin.rep import SurfaceMesh
import random
import thingi10k

import create_results_utils as utils
from sieves import mesh_utils
from sieves import testing


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('mesh_A_filename', type=str)
  parser.add_argument('num_B_meshes', type=int)
  args = parser.parse_args()

  return args


def get_scale(mesh: SurfaceMesh,
              simulation_scale: float):
  '''
  Given a mesh, determine a scale factor that normalizes the mesh,
  scales it by 0.75, and scales it again by some random value in [0.9, 1.1]
  to obtain meshes of similar size that all fit in the simulation space.
  '''

  aabb_min, aabb_max = mesh_utils.get_aabb_corners(mesh)
  max_aabb_side_len = (aabb_max - aabb_min).max().item()
  rand_scale = random.uniform(0.9, 1.1)
  mesh_scale = (1. / simulation_scale) / max_aabb_side_len * 0.75 * rand_scale

  return mesh_scale


def main(args):
  mesh_A_filename = args.mesh_A_filename
  num_B_meshes = args.num_B_meshes

  simulation_scale = 0.01

  # Get scale for mesh A.
  mesh_A_filepath = os.path.join(DATA_DIR, mesh_A_filename)
  mesh_A = mesh_utils.import_mesh(mesh_A_filepath)
  mesh_A_scale = get_scale(mesh_A, simulation_scale)
  meshes_A_configs = [utils.MeshConfig(mesh_A_filename, mesh_A_scale, 1.0)]

  # Pick a set of random mesh B files without any repeats.
  thingi10k.init()
  mesh_entries = [entry for entry in thingi10k.dataset(num_components=1,
                                                       num_facets=(1000, 60000),
                                                       license='creative commons')]
  random_mesh_B_entries = random.sample(mesh_entries, num_B_meshes)

  mesh_B_filenames = [f'{entry["file_id"]}.stl' for entry in random_mesh_B_entries]
  mesh_B_vertex_face_pairs = [thingi10k.load_file(entry['file_path']) for entry in random_mesh_B_entries]

  # Save meshes to data directory.
  for (V, F), filename in zip(mesh_B_vertex_face_pairs, mesh_B_filenames):
    mesh_utils.export_mesh_as_stl(os.path.join(DATA_DIR, filename), V, F)

  # Get scales for B meshes.
  mesh_B_scales = [get_scale(mesh_utils.import_mesh(os.path.join(DATA_DIR, filename)), simulation_scale)
                   for filename in mesh_B_filenames]
  meshes_B_configs = [utils.MeshConfig(mesh_B_filenames[i], mesh_B_scales[i], None)
                      for i in range(num_B_meshes)]

  # Create config dict.
  mesh_A_name = mesh_A_filename.rstrip().rsplit('.', 1)[0]
  config_name = f'{mesh_A_name}_vs_{num_B_meshes}_random_B_meshes'
  config_dict = utils.create_config_dict(meshes_A_configs, meshes_B_configs, config_name)
  config_dict['simulation_scale'] = simulation_scale
  config_dict['image_shape'] = (128, 128)

  # Create config JSON file.
  config_json_filepath = f'{config_name}.json'
  utils.save_dict_as_json(config_dict, config_json_filepath)

  # Create SLURM job file.
  slurm_job_filename = utils.create_slurm_script(config_name, FILE_DIR, config_json_filepath,
                                                 [('account', 'ostein_1459'),
                                                  ('partition', 'gpu'),
                                                  ('nodes', '1'),
                                                  ('ntasks', '1'),
                                                  ('cpus-per-task', '2'),
                                                  ('gpus-per-task', 'a40:1'),
                                                  ('mem', '16GB'),
                                                  ('time', '24:00:00')])

  print(f'Created config file {config_json_filepath} and SLURM job file {slurm_job_filename}.')

  return


if __name__ == '__main__':
  testing.set_determinism(random_seed=15)
  args = parse_args()
  main(args)
