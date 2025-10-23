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

  # Render sieve.
  render.init_blender(res, num_samples)
  render.render_sieve_mesh(result_name, sieve_pos=(1.5, 0.0, 0.0))

  # Render all A and B meshes in orthographic view.
  render.render_A_and_B_meshes_orthographic_views(result_name,
                                                  config_dict,
                                                  result_dict,
                                                  res,
                                                  num_samples)

  # Render all A meshes in perspective view.
  render.render_player_meshes(result_name, config_dict['meshes_A'],
                              render.mesh_color_A,
                              [(1., 0., 0.27)],
                              [(0, 0, 140)],
                              res=res,
                              num_samples=num_samples,
                              is_A=True)

  # Render all B meshes in perspective view.
  render.render_player_meshes(result_name, config_dict['meshes_B'],
                              render.mesh_color_B,
                              [(1., 0., 0.25), (1., 0., 0.31)],
                              [(0, 0, 140), (42, 0, 140)],
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
                                [0.38],
                                0.6,
                                None,
                                None,
                                res,
                                num_samples,
                                True,
                                cam_location=(1.8, 0., 3.),
                                cam_rotation=(30, 0, 90))

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
                                None,
                                None,
                                res,
                                num_samples,
                                False,
                                cam_location=(1.8, 0., 3.),
                                cam_rotation=(30, 0, 90))

  # Render rasters
  render.render_rasters(result_name, config_dict, result_dict)

  return


if __name__ == '__main__':
  result_name = 'wolf_multi_B'
  args = render.parse_args()
  main(args, result_name)
