from collections import namedtuple
import json
import os


MeshConfig = namedtuple('MeshConfig',
                        ['filename', 'fabrication_scale', 'fabrication_offset'])


def create_config_dict(meshes_A_configs: list[MeshConfig],
                       meshes_B_configs: list[MeshConfig],
                       name: str):
  '''
  Create result configuration dictionary with default values.
  '''

  meshes_A = [{'filename': mesh_A_config.filename,
               'fabrication_scale': mesh_A_config.fabrication_scale,
               'fabrication_offset': mesh_A_config.fabrication_offset}
               for mesh_A_config in meshes_A_configs]

  meshes_B = [{'filename': mesh_B_config.filename,
               'fabrication_scale': mesh_B_config.fabrication_scale,
               'fabrication_offset': mesh_B_config.fabrication_offset}
               for mesh_B_config in meshes_B_configs]

  config_dict = {
    "name": name,
    "simulation_scale": 0.01,
    "meshes_A": meshes_A,
    "meshes_B": meshes_B,
    "meshes_A_winning_rot_vecs": None,
    "mesh_A_opt_num_posns": 10,
    "mesh_A_fill_soft_mask_holes": True,
    "mesh_B_opt_num_posns": 10,
    "mesh_B_opt_posn_trans_bounds": [-0.1, 0.1],
    "mesh_B_opt_use_area_proportion": True,
    "mesh_B_opt_vtx_out_of_bounds_loss_weight": 1000.0,
    "mesh_B_opt_iters": 100,
    "mesh_B_opt_learning_rate": 0.05,
    "mesh_A_pso_c1": 0.25,
    "mesh_A_pso_c2": 0.25,
    "mesh_A_pso_w": 0.5,
    "mesh_A_pso_iters": 10,
    "proj_A_pso_c1": 0.05,
    "proj_A_pso_c2": 0.05,
    "proj_A_pso_w": 0.5,
    "proj_A_pso_iters": 10,
    "pso_verbose": True,
    "image_shape": [256, 256],
    "sieve_side_length": 80.0,
    "sieve_thickness": 60.0,
    "sieve_resolution": [1024, 1024]
  }

  return config_dict


def save_dict_as_json(dict: dict,
                      filepath: str):
  '''
  Save a dictionary as a JSON file.
  '''

  with open(filepath, 'w') as outfile:
    json.dump(dict, outfile, indent=2)

  return


def create_slurm_script(job_name: str,
                        job_filedir: str,
                        config_json_filepath: str,
                        slurm_options: list[tuple[str, str]] = [
                          ('account', 'ostein_1459'),
                          ('partition', 'gpu'),
                          ('nodes', '1'),
                          ('ntasks', '1'),
                          ('cpus-per-task', '2'),
                          ('gpus-per-task', 'a40:1'),
                          ('mem', '16GB'),
                          ('time', '6:00:00')
                        ]):
  '''
  Create a SLURM job file.
  '''

  job_filename = f'{job_name}.job'
  job_filepath = os.path.join(job_filedir, job_filename)

  job_specific_slurm_options = [
    ('job-name', job_name),
    ('output', f'{job_name}.out')
  ]

  with open(job_filepath, 'w') as f:
    f.write('#!/bin/bash\n\n')

    # Setup SLURM options.
    for option, value in slurm_options:
      f.write(f'#SBATCH --{option}={value}\n')

    for option, value in job_specific_slurm_options:
      f.write(f'#SBATCH --{option}={value}\n')

    f.write('\n')

    # Setup modules.
    f.write('module purge\n\n')

    # Setup conda environment.
    f.write('eval "$(conda shell.bash hook)"\n')
    f.write('conda activate sieves-kaolin\n\n')

    # Add line for running the actual program.
    f.write(f'python /home1/jaeyoonc/sieves-kaolin/scripts/generate_sieve.py \\\n')
    f.write(f'       {config_json_filepath} \\\n')
    f.write(f'       /home1/jaeyoonc/sieves-kaolin/results/{job_name}')

  return job_filename
