import os

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import create_results_utils as utils


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('config_filepath', type=str)
  parser.add_argument('param', type=str)
  parser.add_argument('param_values', nargs='+', type=int)
  args = parser.parse_args()

  return args


def create_config_and_job_files(config_dict: dict,
                                param: str,
                                param_values: list):
  '''
  Create JSON and SLURM job files for a result with different parameter values.
  '''

  base_name = config_dict['name']
  job_filenames = [None for _ in param_values]

  for i, param_value in enumerate(param_values):
    modified_config_dict = config_dict.copy()
    modified_config_dict_name = f'{base_name}_{param}_{param_value}'
    modified_config_dict['name'] = modified_config_dict_name
    modified_config_dict[param] = param_value

    config_json_filepath = os.path.join(FILE_DIR, f'{modified_config_dict_name}.json')

    # Create config JSON file.
    utils.save_dict_as_json(modified_config_dict, config_json_filepath)

    # Create SLURM job file.
    job_filename = utils.create_slurm_script(modified_config_dict_name, FILE_DIR, config_json_filepath)

    # Save job name.
    job_filenames[i] = job_filename

  return job_filenames


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
  config_filepath = args.config_filepath
  param = args.param
  param_values = args.param_values

  with open(config_filepath, 'r') as config_file:
    config_dict = json.load(config_file)

  job_filenames = create_config_and_job_files(config_dict, param, param_values)

  list_filename = f'{config_dict["name"]}_{param}_jobs.txt'
  save_job_filenames(list_filename, job_filenames)

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
