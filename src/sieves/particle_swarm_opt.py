from collections.abc import Callable
import torch

class ParticleSwarmOptimizer():
  def __init__(self,
               num_particles: int,
               dim: int,
               c1: float,
               c2: float,
               w: float,
               init_pos: torch.FloatTensor,
               obj_func: Callable,
               obj_func_kwargs: dict):
    '''
    Constructor for a particle swarm optimizer that tries to maximize an objective value.

    Input:
    - num_particles: int
        Number of particles to use.

    - dim: int
        Dimension of each particle.

    - c1: float
        Cognitive coefficient.

    - c2: float
        Social coefficient.

    - w: float
        Inertia weight.

    - init_pos: torch.FloatTensor
        Initial particle values.

    - obj_func: Callable
        Function that computes the objective value. Its first parameter must accept a tensor of particles.

    - obj_func_kwargs: dict
        Dictionary of keyword arguments to pass to `obj_func`.
    '''

    # Set up PSO parameters.
    self.num_particles = num_particles
    self.dim = dim
    self.c1 = c1
    self.c2 = c2
    self.w = w

    # Set initial particle positions.
    self.pos = init_pos.detach().clone()

    # Set and evaluate objective function.
    self.obj_func = obj_func
    self.obj_func_kwargs = obj_func_kwargs
    self.evaluate_obj_func()

    # Initialize variables tracking the best values.
    curr_max_idx = torch.argmax(self.curr_val_per_particle)
    self.best_pos = self.pos[curr_max_idx].detach().clone()
    self.best_val = self.curr_val_per_particle[curr_max_idx]
    self.best_pos_per_particle = self.pos.detach().clone()
    self.best_val_per_particle = self.curr_val_per_particle.detach().clone()

    # Initialize velocity.
    self.init_velocity()

    return


  def evaluate_obj_func(self):
    '''
    Evaluates objective function and updates the current values per particle.
    '''

    self.curr_val_per_particle = self.obj_func(self.pos, **self.obj_func_kwargs)

    return
  

  def update_best_vars(self):
    '''
    Update the best global and per particle positions seen so far.
    '''

    # Update global best.
    curr_max_idx = torch.argmax(self.curr_val_per_particle)
    curr_max_val = self.curr_val_per_particle[curr_max_idx]

    if curr_max_val > self.best_val:
      self.best_val = curr_max_val
      self.best_pos = self.pos[curr_max_idx].detach().clone()
    
    # Update per particle best.
    curr_is_greater = self.curr_val_per_particle > self.best_val_per_particle
    self.best_val_per_particle = torch.where(curr_is_greater, self.curr_val_per_particle, self.best_val_per_particle)
    self.best_pos_per_particle = torch.where(curr_is_greater.unsqueeze(1), self.pos, self.best_pos_per_particle)

    return

  
  def init_velocity(self,
                    min_vel: float = 0.,
                    max_vel: float = 1.):
    '''
    Set random initial particle velocities.

    Input:
    - min_vel: float, optional (default 0.0)
        Lower bound for initial velocities.

    - max_vel: float, optional (default 1.0)
        Upper bound for initial velocities.
    '''

    self.velocity = (max_vel - min_vel) * torch.rand(size=(self.num_particles, self.dim)).cuda() + min_vel
  
    return
  

  def update_velocity(self):
    '''
    Update each particle's velocities.
    '''

    rand_1 = torch.rand(size=(self.num_particles, self.dim)).cuda()
    rand_2 = torch.rand(size=(self.num_particles, self.dim)).cuda()

    self.velocity = self.w * self.velocity + \
                    self.c1 * rand_1 * (self.best_pos_per_particle - self.pos) + \
                    self.c2 * rand_2 * (self.best_pos - self.pos)

    return
  

  def update_position(self):
    '''
    Update each particle's position.
    '''
    
    self.pos += self.velocity

    return
  

  def optimize(self,
               num_iters: int,
               verbose: bool = False):
    '''
    Run particle swarm optimization.

    Input:
    - num_iters: int
        Number of iterations of PSO to run.

    - verbose: bool, optional (default False)
        Flag to print verbose output during optimization.

    Output:
    - best_val: float
        Best objective value observed across all iterations.

    - best_pos: torch.Tensor
        Particle position corresponding to `best_val`.
    '''

    for iter in range(num_iters):
      self.update_velocity()
      self.update_position()
      self.evaluate_obj_func()
      self.update_best_vars()

      if verbose:
        print(f'Iteration {iter + 1}/{num_iters}, best value: {self.best_val.item()}, best position: {self.best_pos}')

    return self.best_val, self.best_pos
