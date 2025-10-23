import numpy as np
import os
import random
import torch


def set_determinism(random_seed: int = 0,
                    torch_seed: int = 0,
                    numpy_seed: int = 0):
  '''
  Enable determinism for various packages.

  Input:
  - random_seed: int, optional (default 0)
      Seed to use for the Python package `random`.

  - torch_seed: int, optional (default 0)
      Seed to use for PyTorch.

  - numpy_seed: int, optional (default 0)
      Seed to use for Numpy.
  '''

  # For Python random
  random.seed(random_seed)

  # For CUDA, see https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  os.environ['PYTHONHASHSEED'] = str(0)

  # For PyTorch, see https://pytorch.org/docs/stable/notes/randomness.html
  torch_seed = 0
  torch.manual_seed(torch_seed)
  torch_cuda_seed = 0
  torch.cuda.manual_seed(torch_cuda_seed)
  torch.use_deterministic_algorithms(True)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  # For NumPy
  numpy_seed = 0
  np.random.seed(numpy_seed)

  return
