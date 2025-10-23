import copy
from kaolin.rep import SurfaceMesh

from sieves import batched_opt as bopt


def find_B_losing_scale(mesh_A: SurfaceMesh,
                        mesh_B: SurfaceMesh,
                        lose_threshold: float = 0.05,
                        lose_init_scale: float = 1.,
                        lose_scale_multiplier: float = 1.5,
                        max_iters: int = 10,
                        pso_kwargs: dict = {
                          'num_A_orientations': 10,
                          'num_B_orientations': 10,
                          'mesh_A_fill_soft_mask_holes': False,
                          'mesh_B_opt_pos_trans_bounds': (-0.05, 0.05),
                          'mesh_B_opt_use_area_proportion': True,
                          'mesh_B_opt_vtx_out_of_bounds_loss_weight': 1000.,
                          'mesh_B_opt_iters': 100,
                          'mesh_B_opt_learning_rate': 0.05,
                          'pso_c1': 0.25,
                          'pso_c2': 0.25,
                          'pso_w': 0.5,
                          'pso_iters': 10,
                          'pso_verbose': False
                        }):
  '''
  Find a scale for mesh B large enough so that it loses against mesh A.

  Input:
  - mesh_A: kaolin.rep.SurfaceMesh
      Mesh to optimize orientation of so that its projection blocks mesh B.

  - mesh_B: kaolin.rep.SurfaceMesh
      Mesh to be blocked.

  - lose_threshold: float, optional (default 0.05)
      Minimum objective value for PSO to attain to be able to say that mesh A blocks mesh B.

  - lose_init_scale: float, optional (default 1.0)
      Initial scale for mesh B.

  - lose_scale_multiplier: float, optional (default 1.5)
      Factor to multiply current scale of mesh B by if it was not a losing scale.

  - max_iters: int, optional (default 10)
      Max number of iterations for increasing the scale of mesh B before quitting.

  - pso_kwargs: dict, optional (default above)
      Arguments for particle swarm optimizer.

  Output:
  - lose_scale: float
      A scale at which mesh B loses against mesh A, not necessarily optimal.

  - best_cost: float
      Objective value of optimizing mesh A against mesh B at the computed scale.

  - best_pos: torch.Tensor of shape (6,)
      Orientation of mesh A that blocks mesh B at the computed scale.
  '''

  lose_mesh =  copy.deepcopy(mesh_B)
  lose_mesh.vertices[:] *= lose_init_scale
  lose_scale = lose_init_scale

  iters = 0
  while iters < max_iters:
    best_cost, best_pos = bopt.optimize_posn_of_A(mesh_A,
                                                  [lose_mesh],
                                                  **pso_kwargs)
    
    if best_cost > lose_threshold:
      break
  
    lose_scale *= lose_scale_multiplier
    lose_mesh.vertices[:] *= lose_scale_multiplier
    iters += 1

  if iters == max_iters:
    print(f'Max iterations ({max_iters}) reached.')
  
  return lose_scale, best_cost, best_pos


def find_B_winning_scale(mesh_A: SurfaceMesh,
                         mesh_B: SurfaceMesh,
                         win_threshold: float = 0.01,
                         win_init_scale: float = 1.,
                         win_scale_multiplier: float = 0.5,
                         max_iters: int = 10,
                         pso_kwargs: dict = {
                           'num_A_orientations': 10,
                           'num_B_orientations': 10,
                           'mesh_A_fill_soft_mask_holes': False,
                           'mesh_B_opt_pos_trans_bounds': (-0.05, 0.05),
                           'mesh_B_opt_use_area_proportion': True,
                           'mesh_B_opt_vtx_out_of_bounds_loss_weight': 1000.,
                           'mesh_B_opt_iters': 100,
                           'mesh_B_opt_learning_rate': 0.05,
                           'pso_c1': 0.25,
                           'pso_c2': 0.25,
                           'pso_w': 0.5,
                           'pso_iters': 10,
                           'pso_verbose': False
                         }):
  '''
  Find a scale for mesh B small enough so that it wins against mesh A.

  Input:
  - mesh_A: kaolin.rep.SurfaceMesh
      Mesh to optimize orientation of so that its projection blocks mesh B.

  - mesh_B: kaolin.rep.SurfaceMesh
      Mesh to be blocked.

  - win_threshold: float, optional (default 0.01)
      Maximum objective value for PSO to attain to be able to say that mesh A fails to block mesh B.

  - win_init_scale: float, optional (default 1.0)
      Initial scale for mesh B.

  - win_scale_multiplier: float, optional (default 0.5)
      Factor to multiply current scale of mesh B by if it was not a winning scale.

  - max_iters: int, optional (default 10)
      Max number of iterations for increasing the scale of mesh B before quitting.

  - pso_kwargs: dict, optional (default above)
      Arguments for particle swarm optimizer.

  Output:
  - win_scale: float
      A scale at which mesh B wins against mesh A, not necessarily optimal.

  - best_cost: float
      Objective value of optimizing mesh A against mesh B at the computed scale.

  - best_pos: torch.Tensor of shape (6,)
      Orientation of mesh A that attempts to block mesh B at the computed scale.
  '''

  win_mesh =  copy.deepcopy(mesh_B)
  win_mesh.vertices[:] *= win_init_scale
  win_scale = win_init_scale

  iters = 0
  while iters < max_iters:
    best_cost, best_pos = bopt.optimize_posn_of_A(mesh_A,
                                                  [win_mesh],
                                                  **pso_kwargs)
    
    if best_cost < win_threshold:
      break
  
    win_scale *= win_scale_multiplier
    win_mesh.vertices[:] *= win_scale_multiplier
    iters += 1

  if iters == max_iters:
    print(f'Max iterations ({max_iters}) reached.')
  
  return win_scale, best_cost, best_pos


def find_optimal_B_scales(mesh_A: SurfaceMesh,
                          mesh_B: SurfaceMesh,
                          max_iters: int = 20,
                          gap_threshold: float = 0.1,
                          lose_scale_search_threshold: float = 0.05,
                          lose_scale_search_init_scale: float = 1.,
                          lose_scale_search_scale_multiplier: float = 1.5,
                          lose_scale_search_max_iters: int = 10,
                          win_scale_search_threshold: float = 0.01,
                          win_scale_search_init_scale: float = 1.,
                          win_scale_search_scale_multiplier: float = 0.5,
                          win_scale_search_max_iters: int = 10,
                          pso_kwargs: dict = {
                            'num_A_orientations': 10,
                            'num_B_orientations': 10,
                            'mesh_A_fill_soft_mask_holes': False,
                            'mesh_B_opt_pos_trans_bounds': (-0.05, 0.05),
                            'mesh_B_opt_use_area_proportion': True,
                            'mesh_B_opt_vtx_out_of_bounds_loss_weight': 1000.,
                            'mesh_B_opt_iters': 100,
                            'mesh_B_opt_learning_rate': 0.05,
                            'pso_c1': 0.25,
                            'pso_c2': 0.25,
                            'pso_w': 0.5,
                            'pso_iters': 10,
                            'pso_verbose': False
                          },
                          verbose: bool = False):
  '''
  Use binary search to find optimal scales for which mesh B loses and wins against mesh A.

  Input:
  - mesh_A: kaolin.rep.SurfaceMesh
      Mesh to optimize orientation of so that its projection blocks mesh B.

  - mesh_B: kaolin.rep.SurfaceMesh
      Mesh to be blocked.

  - max_iters: int, optional (default 20)
      Max number of iterations for binary search.

  - gap_threshold: float, optional (default 0.1)
      Maximum gap between winning and losing scales of mesh B to attain before quitting binary search.

  - lose_scale_search_threshold: float, optional (default 0.05)
      Minimum objective value for PSO to attain to be able to say that mesh A blocks mesh B.

  - lose_scale_search_init_scale: float, optional (default 1.0)
      Initial scale for mesh B.

  - lose_scale_search_scale_multiplier: float, optional (default 1.5)
      Factor to multiply current scale of mesh B by if it was not a losing scale.

  - lose_scale_search_max_iters: int, optional (default 10)
      Max number of iterations for increasing the scale of mesh B before quitting.

  - win_scale_search_threshold: float, optional (default 0.01)
      Maximum objective value for PSO to attain to be able to say that mesh A fails to block mesh B.

  - win_scale_search_init_scale: float, optional (default 1.0)
      Initial scale for mesh B.

  - win_scale_search_scale_multiplier: float, optional (default 0.5)
      Factor to multiply current scale of mesh B by if it was not a winning scale.

  - win_scale_search_max_iters: int, optional (default 10)
      Max number of iterations for deccreasing the scale of mesh B before quitting.

  - pso_kwargs: dict, optional (default above)
      Arguments for particle swarm optimizer.

  - verbose: bool, optional (default False)
      Option for printing verbose output.

  Output:
  - ub_scale: float
      Scale at which mesh B loses.

  - ub_cost: float
      Objective value returned by PSO from optimizing mesh A against mesh B scaled by `ub_scale`.

  - ub_pos: torch.Tensor of shape (6,)
      Orientation returned by PSO from optimizing mesh A against mesh B scaled by `ub_scale`.

  - lb_scale: float
      Scale at which mesh B loses.

  - lb_cost: float
      Objective value returned by PSO from optimizing mesh A against mesh B scaled by `lb_scale`.

  - lb_pos: torch.Tensor of shape (6,)
      Orientation returned by PSO from optimizing mesh A against mesh B scaled by `lb_scale`.
  '''

  # During the search, ub_scale will always be a scale large enough so that mesh A
  # can find a cost greater than the threshold where B loses.
  ub_scale, ub_cost, ub_pos = find_B_losing_scale(mesh_A=mesh_A,
                                                  mesh_B=mesh_B,
                                                  lose_threshold=lose_scale_search_threshold,
                                                  lose_init_scale=lose_scale_search_init_scale,
                                                  lose_scale_multiplier=lose_scale_search_scale_multiplier,
                                                  max_iters=lose_scale_search_max_iters,
                                                  pso_kwargs=pso_kwargs)
  
  # During the search, lb_scale will always be a scale small enough so that mesh A
  # cannot find a cost greater than the threshold where B loses.
  lb_scale, lb_cost, lb_pos = find_B_winning_scale(mesh_A=mesh_A,
                                                   mesh_B=mesh_B,
                                                   win_threshold=win_scale_search_threshold,
                                                   win_init_scale=win_scale_search_init_scale,
                                                   win_scale_multiplier=win_scale_search_scale_multiplier,
                                                   max_iters=win_scale_search_max_iters,
                                                   pso_kwargs=pso_kwargs)

  if verbose:
    print(f'Iteration 0, lower bound: {lb_scale}, upper bound: {ub_scale}')

  assert(ub_scale >= lb_scale)
  curr_gap = ub_scale - lb_scale
  iter = 0

  while iter < max_iters and curr_gap > gap_threshold:
    mid = 0.5 * (ub_scale + lb_scale)

    scaled_mesh = copy.deepcopy(mesh_B)
    scaled_mesh.vertices[:] *= mid

    best_cost, best_pos = bopt.optimize_posn_of_A(mesh_A,
                                                  [scaled_mesh],
                                                  **pso_kwargs)

    mesh_B_loses = best_cost > lose_scale_search_threshold
    if mesh_B_loses:
      ub_scale = mid
      ub_cost = best_cost
      ub_pos = best_pos
    else:
      lb_scale = mid
      lb_cost = best_cost
      lb_pos = best_pos
    
    curr_gap = ub_scale - lb_scale
    iter += 1

    if verbose:
      print(f'Iteration {iter}, lower bound: {lb_scale}, upper bound: {ub_scale}')
  
  if iter == max_iters:
    print(f'Max iterations ({max_iters}) reached.')

  return ub_scale, ub_cost, ub_pos, lb_scale, lb_cost, lb_pos
