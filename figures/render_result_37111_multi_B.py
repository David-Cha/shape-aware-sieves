import os
import render


def main(args, result_name):
  '''
  Create directory to store all figures for this result.
  '''

  figures_dir = os.path.abspath(os.path.join(render.FILE_DIR, result_name))
  os.makedirs(figures_dir, exist_ok=True)

  '''
  Get Blender settings.
  '''

  res = (args.resx, args.resy)
  num_samples = args.samples

  '''
  Read config and results JSON files.
  '''

  config_dict, result_dict = render.process_config_and_result_dicts(result_name)
  
  '''
  Render all figures.
  '''

  meshes_A_use_edge_splitting_flags = [True]
  meshes_A_use_smooth_shading_flags = [True]
  meshes_B_use_edge_splitting_flags = [False, True]
  meshes_B_use_smooth_shading_flags = [True, True]

  # Render sieve.
  render.init_blender(res, num_samples)
  render.render_sieve_mesh(result_name, cam_location=(4., 0., 3.5))

  # Render all A and B meshes in orthographic view.
  render.render_A_and_B_meshes_orthographic_views(result_name,
                                                  config_dict,
                                                  result_dict,
                                                  res,
                                                  num_samples,
                                                  meshes_A_use_edge_splitting_flags,
                                                  meshes_A_use_smooth_shading_flags,
                                                  meshes_B_use_edge_splitting_flags,
                                                  meshes_B_use_smooth_shading_flags)

  # Render all A meshes in perspective view.
  render.render_player_meshes(result_name, config_dict['meshes_A'],
                              render.mesh_color_A,
                              [(1., 0., 0.43)],
                              [(0, 0, 45)],
                              use_edge_splitting_flags=meshes_A_use_edge_splitting_flags,
                              use_smooth_shading_flags=meshes_A_use_smooth_shading_flags,
                              res=res,
                              num_samples=num_samples,
                              is_A=True)

  # Render all B meshes in perspective view.
  render.render_player_meshes(result_name, config_dict['meshes_B'],
                              render.mesh_color_B,
                              [(1., 0., 0.56), (1., 0., 0.36)],
                              [(0, 0, 120), (0, 0, 120)],
                              use_edge_splitting_flags=meshes_B_use_edge_splitting_flags,
                              use_smooth_shading_flags=meshes_B_use_smooth_shading_flags,
                              res=res,
                              num_samples=num_samples,
                              is_A=False)

  # Render all A meshes with sieve.
  render.render_meshes_in_sieve(result_name,
                                config_dict['simulation_scale'],
                                render.mesh_color_A,
                                config_dict['meshes_A'],
                                result_dict['meshes_A_rot_vecs'],
                                result_dict['meshes_A_trans_vecs'],
                                None,
                                [0.27],
                                0.6,
                                meshes_A_use_edge_splitting_flags,
                                meshes_A_use_smooth_shading_flags,
                                res,
                                num_samples,
                                True,
                                cam_location=(2., 0., 4.),
                                cam_rotation=(25, 0, 90))

  # Render all B meshes with sieve.
  render.render_meshes_in_sieve(result_name,
                                config_dict['simulation_scale'],
                                render.mesh_color_B,
                                config_dict['meshes_B'],
                                result_dict['meshes_B_rot_vecs'],
                                result_dict['meshes_B_trans_vecs'],
                                result_dict['meshes_B_best_losses'],
                                None,
                                None,
                                meshes_B_use_edge_splitting_flags,
                                meshes_B_use_smooth_shading_flags,
                                res,
                                num_samples,
                                False,
                                cam_location=(2., 0., 4.),
                                cam_rotation=(25, 0, 90))

  # Render rasters
  render.render_rasters(result_name, config_dict, result_dict)

  return


if __name__ == '__main__':
  result_name = '37111_multi_B'
  args = render.parse_args()
  main(args, result_name)
