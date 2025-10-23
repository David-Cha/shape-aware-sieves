import os
import sys

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'data'))
PKG_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, 'src'))
sys.path.insert(1, PKG_DIR)

from matplotlib import pyplot as plt
import torch

from sieves import batched_opt as bopt
from sieves import mesh_utils


def main():
  mesh_filename = '67223.stl'
  mesh = mesh_utils.import_mesh(os.path.abspath(os.path.join(DATA_DIR, mesh_filename)),
                                fabrication_scale=0.4,
                                fabrication_offset=None,
                                simulation_scale=0.01)
  mesh_transformed = mesh_utils.get_transformed_mesh(mesh,
                                                     torch.tensor([1.0554171800613403,
                                                                   1.0557841062545776,
                                                                   0.2520529329776764,
                                                                   1.0068082809448242,
                                                                   1.005874752998352,
                                                                   0.3791911005973816]).cuda())
  image_shape = (2048, 2048)

  soft_mask = bopt.get_soft_masks(mesh_transformed.vertices.unsqueeze(0), mesh.faces, image_shape, False)[0]

  plt.figure(figsize=(2.048, 2.048))
  plt.axis('off')
  plt.imshow(soft_mask.detach().clone().cpu().numpy())
  cbar = plt.colorbar(shrink=0.7)
  cbar.ax.tick_params(labelsize=6)
  plt.savefig(os.path.join(FILE_DIR, f'example_soft_mask.png'),
              transparent=True,
              dpi=1000)

if __name__ == '__main__':
  main()
