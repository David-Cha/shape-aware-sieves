import os
from matplotlib import pyplot as plt
import torch

import render
from sieves import batched_opt as bopt
from sieves import mesh_utils
from sieves import orientation as orie
from sieves import visualization as vis


def main(args, result_name):
  '''
  Get optimized position of each mesh B.
  '''

  config_dict, result_dict = render.process_config_and_result_dicts(result_name)

  simulation_scale = config_dict['simulation_scale']

  meshes_A = [mesh_utils.import_mesh_from_config(mesh_A_config, render.DATA_DIR, simulation_scale) for mesh_A_config in config_dict['meshes_A']]
  meshes_B = [mesh_utils.import_mesh_from_config(mesh_B_config, render.DATA_DIR, simulation_scale) for mesh_B_config in config_dict['meshes_B']]

  # Get the union mesh.
  meshes_A_rot_vecs = torch.tensor(result_dict['meshes_A_rot_vecs']).cuda()
  meshes_A_trans_vecs = torch.tensor(result_dict['meshes_A_trans_vecs']).cuda()
  meshes_A_transformed = [mesh_utils.get_transformed_mesh(mesh_A, meshes_A_rot_vecs[i], meshes_A_trans_vecs[i])
                          for i, mesh_A in enumerate(meshes_A)]
  union_mesh = mesh_utils.get_union_mesh(meshes_A_transformed)

  # Get optimized positions of B meshes.
  mesh_B_opt_num_posns = config_dict['mesh_B_opt_num_posns']

  rot_vec_per_B_posn = orie.generate_rotation_vectors(mesh_B_opt_num_posns)
  trans_vec_per_B_posn = orie.generate_uniform_random_vectors(mesh_B_opt_num_posns, 2, *config_dict['mesh_B_opt_posn_trans_bounds'])

  rot_vec_per_A_posn = torch.tensor([[1., 0., 0., 0., 1., 0.]]).cuda() # identity rotation vector

  num_B_meshes = len(meshes_B)
  repeat_shape = [num_B_meshes, 1]
  rot_vec_per_B_posn_per_A_posn_per_B_mesh = rot_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(rot_vec_per_B_posn.shape))])
  trans_vec_per_B_posn_per_A_posn_per_B_mesh = trans_vec_per_B_posn.unsqueeze(0).repeat(repeat_shape + [1 for _ in range(len(trans_vec_per_B_posn.shape))])

  _, opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh, opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh = bopt.optimize_B_posn_per_A_posn_per_B_mesh(rot_vec_per_A_posn=rot_vec_per_A_posn,
                                                                                                                                               mesh_A=union_mesh,
                                                                                                                                               meshes_B=meshes_B,
                                                                                                                                               rot_vec_per_B_posn_per_A_posn_per_B_mesh=rot_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                                                                               trans_vec_per_B_posn_per_A_posn_per_B_mesh=trans_vec_per_B_posn_per_A_posn_per_B_mesh,
                                                                                                                                               mesh_A_fill_soft_mask_holes=config_dict['mesh_A_fill_soft_mask_holes'],
                                                                                                                                               mesh_B_opt_use_area_proportion=config_dict['mesh_B_opt_use_area_proportion'],
                                                                                                                                               mesh_B_opt_vtx_out_of_bounds_loss_weight=config_dict['mesh_B_opt_vtx_out_of_bounds_loss_weight'],
                                                                                                                                               mesh_B_opt_iters=config_dict['mesh_B_opt_iters'],
                                                                                                                                               mesh_B_opt_learning_rate=config_dict['mesh_B_opt_learning_rate'],
                                                                                                                                               image_shape=config_dict['image_shape'])

  '''
  Create directory to store all figures for this result.
  '''

  dir_name = 'many_B_orientations'
  figures_dir = os.path.abspath(os.path.join(render.FILE_DIR, dir_name))
  os.makedirs(figures_dir, exist_ok=True)

  '''
  Render rasters.
  '''

  for mesh_idx, mesh_B in enumerate(meshes_B):
    mesh_B_config = config_dict['meshes_B'][mesh_idx]
    mesh_B_filename = mesh_B_config['filename']
    mesh_B_name = mesh_B_filename.rsplit('.', 1)[0]

    opt_rot_vecs = opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh[mesh_idx][0]
    opt_trans_vecs = opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh[mesh_idx][0]

    for posn_idx in range(mesh_B_opt_num_posns):
      mesh_B_transformed = mesh_utils.get_transformed_mesh(mesh_B, opt_rot_vecs[posn_idx], opt_trans_vecs[posn_idx])
      figure_raster = vis.get_figure_raster(union_mesh, mesh_B_transformed, torch.IntTensor(render.color_B).cuda())

      plt.figure(figsize=(2.048, 2.048))
      plt.axis('off')
      plt.imshow(figure_raster)
      plt.savefig(os.path.join(figures_dir, f'mesh_B_{mesh_B_name}_posn_{posn_idx}_raster.png'),
                  transparent=True,
                  dpi=1000)

      print(f'Rendered raster of mesh {mesh_idx} at position {posn_idx}.')


  '''
  Render mesh blocked by sieve.
  '''

  res = (args.resx, args.resy)
  num_samples = args.samples

  for mesh_idx, mesh_B in enumerate(meshes_B):
    opt_rot_vecs = opt_rot_vec_per_B_posn_per_A_posn_per_B_mesh[mesh_idx][0].tolist()
    opt_trans_vecs = opt_trans_vec_per_B_posn_per_A_posn_per_B_mesh[mesh_idx][0].tolist()

    mesh_B_config = config_dict['meshes_B'][mesh_idx]
    mesh_B_filename = mesh_B_config['filename']
    mesh_B_name = mesh_B_filename.rsplit('.', 1)[0]

    mesh_configs = [mesh_B_config for _ in range(mesh_B_opt_num_posns)]
    mesh_blocked = [True for _ in range(mesh_B_opt_num_posns)]
    z_offsets_without_sieve_offset = render.compute_offsets_for_meshes_in_sieve(result_name,
                                                                                simulation_scale,
                                                                                mesh_configs,
                                                                                opt_rot_vecs,
                                                                                opt_trans_vecs,
                                                                                mesh_blocked)
    blender_scale = 0.01
    sieve_pos_z = 0.3
    z_offsets = [offset * blender_scale + sieve_pos_z for offset in z_offsets_without_sieve_offset]

    mesh_use_edge_splitting_flags = [True for _ in range(mesh_B_opt_num_posns)]
    mesh_use_smooth_shading_flags = [True for _ in range(mesh_B_opt_num_posns)]

    for posn_idx in range(mesh_B_opt_num_posns):
      render.init_blender(res, num_samples)
      render.render_mesh_in_sieve(result_name=result_name,
                                  sim_scale=simulation_scale,
                                  mesh_filename=mesh_B_filename,
                                  mesh_color=render.mesh_color_B,
                                  mesh_fab_scale=mesh_B_config['fabrication_scale'],
                                  mesh_rot_vec=opt_rot_vecs[posn_idx],
                                  mesh_trans_vec=opt_trans_vecs[posn_idx],
                                  mesh_pos_z=z_offsets[posn_idx],
                                  mesh_use_edge_splitting=mesh_use_edge_splitting_flags[posn_idx],
                                  mesh_use_smooth_shading=mesh_use_smooth_shading_flags[posn_idx],
                                  sieve_color=render.sieve_color,
                                  sieve_pos_z=sieve_pos_z,
                                  scale=blender_scale,
                                  save_filepath=os.path.join(figures_dir, f'mesh_B_{mesh_B_name}_posn_{posn_idx}_in_sieve'))

      print(f'Rendered sieve image of mesh {mesh_idx} at position {posn_idx}.')

  return


if __name__ == '__main__':
  args = render.parse_args()
  result_name = '38631_vs_39880'
  main(args, result_name)
