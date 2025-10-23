from scipy.stats import special_ortho_group
import torch


def generate_uniform_random_vectors(num_vectors: int,
                                    vector_len: int,
                                    lb: float,
                                    ub: float):
  '''
  Generate vectors with random values from a uniform distribution.

  Input:
  - num_vectors: int
      Number of vectors to generate.

  - vector_len: int
      Size of each vector.

  - lb: float
      Lower bound for random values.

  - ub: float
      Upper bound for random values.

  Output:
  - random_vectors: torch.Tensor of shape (num_vectors, vector_len)
      Vectors of random values in [lb, ub] generated from a uniform distribution.
  '''

  return (ub - lb) * torch.rand(num_vectors, vector_len).cuda() + lb


def generate_rotation_vectors(num_vectors: int):
  '''
  Generate random 6D rotation vectors.

  Input:
  - num_vectors: int
      Number of rotation vectors to generate.

  Output:
  - rot_vecs: torch.Tensor of shape (num_vectors, 6)
      Randomly generated 6D rotation vectors.
  '''

  rot_mats = special_ortho_group.rvs(3, num_vectors)
  rot_vecs = rot_mats[..., :2, :].reshape((num_vectors, 6))

  return torch.FloatTensor(rot_vecs).cuda()


def get_rot_mats(rot_vecs: torch.tensor):
  '''
  Convert a batch of 6D rotation vectors to rotation matrices.

  Input:
  - rot_vecs: torch.Tensor of shape (..., 6)
      Batch of 6D vectors representing rotations based on the paper
      "On the Continuity of Rotation Representations in Neural Networks"
      by Zhou et al. from CVPR 2019.

  Output:
  - rot_mats: torch.Tensor of shape (..., 3, 3)
      Batch of rotation matrices corresponding to the rotation vectors.
  '''

  v1 = rot_vecs[...,0:3]
  v2 = rot_vecs[...,3:6]
  
  r1 = torch.nn.functional.normalize(v1, dim=-1)
  r2 = torch.nn.functional.normalize(v2 - (r1 * v2).sum(-1, keepdim=True) * r1, dim=-1)
  r3 = torch.linalg.cross(r1, r2, dim=-1)
  
  rot_mats = torch.transpose(torch.stack((r1, r2, r3), dim=-2), -2, -1)
  
  return rot_mats


def get_rotated_vtxs(vtxs: torch.Tensor,
                     rot_vecs: torch.Tensor):
  '''
  Get batch of vertices resulting from applying each rotation.

  Input:
  - vtxs: torch.Tensor of shape (V, 3)
      Coordinates of a list of vertices.

  - rot_vecs: torch.Tensor of shape (..., 6)
      Rotation vectors to transform the vertices by.

  Output:
  - rotated_vtxs: torch.Tensor of shape (..., V, 3)
      Batch of vertices from applying each rotation.
  '''

  rot_mats = get_rot_mats(rot_vecs)
  rotated_vtxs = torch.transpose(torch.matmul(rot_mats, torch.transpose(vtxs, -2, -1)), -2, -1)

  return rotated_vtxs


def get_transformed_vtxs(vtxs: torch.Tensor,
                         rot_vecs: torch.Tensor,
                         trans_vecs: torch.Tensor):
  '''
  Get batch of vertices resulting from applying each rotation and translation pair.

  Input:
  - vtxs: torch.Tensor of shape (V, 3)
      Coordinates of a list of vertices.

  - rot_vecs: torch.Tensor of shape (..., 6)
      Rotation vectors to transform the vertices by.

  - trans_vecs: torch.Tensor of shape (..., 2)
      Translation vectors to transform the vertices by after rotation.

  Output:
  - transformed_vtxs: torch.Tensor of shape (..., V, 3)
      Batch of vertices resulting from applying each rotation and translation pairs.
  '''
  
  assert(rot_vecs.shape[:-1] == trans_vecs.shape[:-1])
  
  trans_vecs_3d = torch.nn.functional.pad(input=trans_vecs, pad=(0,1), mode='constant', value=0.).unsqueeze(-2)
  transformed_vtxs = get_rotated_vtxs(vtxs, rot_vecs) + trans_vecs_3d

  return transformed_vtxs


def get_z_axis_rot_mats(angles: torch.Tensor):
  '''
  Get 2D rotation matrices representing rotation around the z-axis from a list of angles.

  Input:
  - angles: torch.Tensor of shape (n,)
      Batch of angles in radians.

  Output:
  - rot_mats: torch.Tensor of shape (n, 2, 2)
      Batch of rotation matrices corresponding to the rotation angles.
  '''
  cos_angles = torch.cos(angles)
  sin_angles = torch.sin(angles)
  
  rot_mats = torch.zeros(len(angles), 2, 2).cuda()
  rot_mats[:,0,0] = cos_angles
  rot_mats[:,0,1] = -sin_angles
  rot_mats[:,1,0] = sin_angles
  rot_mats[:,1,1] = cos_angles

  return rot_mats


def get_transformed_vtxs_2D(vtxs: torch.Tensor,
                            angles: torch.Tensor,
                            translations: torch.Tensor):
  '''
  Transform 2D vertices by rotation around the z-axis and then by a 2D translation.

  Input:
  - vtxs: torch.Tensor of shape (V, 2)
      2D vertex coordinates.
  
  - angles: torch.Tensor of shape (n,)
      Batch of angles for rotating the vertices.

  - translations: torch.Tensor of shape (n, 2)
      Batch of vectors for translating the vertices.

  Output:
  - vtxs_transformed: torch.Tensor of shape (n, V, 2)
      Batch of vertices resulting from applying the angle and translation vector pairs.
  '''

  rot_mats = get_z_axis_rot_mats(angles)
  vtxs_transformed = torch.transpose(torch.matmul(rot_mats, torch.transpose(vtxs, -2, -1)), -2, -1) + translations.unsqueeze(-2)

  return vtxs_transformed
