import argparse
import os
import subprocess


def parse_args():
  '''
  Parse command line arguments.
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('list_filepath', help='filepath to text file containing list of job filenames', type=str)
  args = parser.parse_args()

  return args


def run_jobs(job_filenames: list[str],
             job_dir: str):
  '''
  Submit a list of SLURM jobs.
  '''

  for job_filename in job_filenames:
    job_filepath = os.path.join(job_dir, job_filename)
    subprocess.run(['sbatch', job_filepath], check=True)

  return


def main(args):
  list_filepath = args.list_filepath
  list_filedir = os.path.abspath(os.path.dirname(os.path.abspath(list_filepath)))

  with open(list_filepath) as file:
    job_filenames = [line.rstrip() for line in file]
  
  run_jobs(job_filenames, list_filedir)

  return


if __name__ == '__main__':
  args = parse_args()
  main(args)
