import os

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('results_list_filepath', type=str)
  args = parser.parse_args()

  return args


def create_slurm_script(job_name: str,
                        job_filedir: str,
                        slurm_options: list[tuple[str, str]] = [
                          ('account', 'ostein_1459'),
                          ('partition', 'gpu'),
                          ('nodes', '1'),
                          ('ntasks', '1'),
                          ('cpus-per-task', '8'),
                          ('gpus-per-task', 'a40:1'),
                          ('mem', '16GB'),
                          ('time', '3:00:00')
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
    f.write('module purge\n')
    f.write('module load gcc libxkbcommon libxi mesa\n\n')

    # Setup conda environment.
    f.write('eval "$(conda shell.bash hook)"\n')
    f.write('conda activate sieves-kaolin\n\n')

    # Add line for running the actual program.
    f.write(f'python /home1/jaeyoonc/sieves-kaolin/figures/{job_name}.py\n')

  return job_filename


def save_job_filenames(list_filename: str,
                       job_filenames: list[str]):
  '''
  Save the list of SLURM job filenames with one per line in a text file.
  '''

  list_filepath = os.path.join(FILE_DIR, list_filename)

  with open(list_filepath, 'w') as f:
    for job_filename in job_filenames:
      f.write(f'{job_filename}\n')

  return


def main(args):
  results_list_filepath = args.results_list_filepath

  with open(results_list_filepath) as file:
    result_names = [line.rstrip() for line in file]
  
  job_filenames = [None for _ in result_names]

  for idx, result_name in enumerate(result_names):
    job_filenames[idx] = create_slurm_script(f'render_result_{result_name}', FILE_DIR)

  list_name = Path(results_list_filepath).stem
  jobs_list_filename = f'{list_name}_jobs.txt'
  save_job_filenames(jobs_list_filename, job_filenames)

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
