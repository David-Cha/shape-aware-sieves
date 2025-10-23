import cv2 as cv
import numpy as np
import torch


def get_filled_rasters(rasters: torch.Tensor):
  '''
  Given a batch of rasters, return a corresponding batch of rasters with all holes filled.

  Input:
  - rasters: torch.Tensor of shape (a, x, y)
      Batch of rasters.

  Output:
  - filled_rasters: torch.Tensor of shape (a, x, y)
      Copy of `rasters` with all holes filled with a value of 1.
  '''

  # Convert to numpy array of type uint8 for processing with OpenCV and have all rasters
  # flattened into one for batch processing.
  raster_flattened_np_uint8 = torch.flatten(rasters, start_dim=0, end_dim=1).detach().clone().cpu().numpy().astype(np.uint8)

  # Get the outermost contours.
  contours, _ = cv.findContours(raster_flattened_np_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

  # Get rasters with all holes filled by filling the contours.
  filled_flattened_np = np.zeros_like(raster_flattened_np_uint8)
  ctr_idx = -1
  thickness = -1
  cv.drawContours(filled_flattened_np, contours, ctr_idx, (1), thickness)

  # Unflatten rasters.
  filled_rasters_flattened = torch.from_numpy(filled_flattened_np).cuda()
  filled_rasters = torch.unflatten(filled_rasters_flattened, 0, rasters.shape[:2])

  return filled_rasters


def get_filled_soft_masks(soft_masks: torch.Tensor):
  '''
  Given a batch of soft masks, return a corresponding batch of soft masks with all holes filled.

  Input:
  - soft_masks: torch.Tensor of shape (a, x, y)
      Batch of soft masks.

  Output:
  - filled_soft_masks: torch.Tensor of shape (a, x, y)
      Copy of `soft_masks` with all holes filled with a value of 1.0, including
      the soft boundaries of the holes.
  '''

  # Get a raster of 0's and 1's from the soft mask by setting all values less than 1 to 0.
  soft_masks_thresholded = soft_masks.detach().clone()
  soft_masks_thresholded[soft_masks_thresholded < 1.] = 0.
  assert(torch.all((soft_masks_thresholded == 0.) | (soft_masks_thresholded == 1.)))

  # Convert to numpy array of type uint8 for processing with OpenCV and have all rasters
  # flattened into one for batch processing.
  raster_flattened_np_uint8 = torch.flatten(soft_masks_thresholded, start_dim=0, end_dim=1).detach().clone().cpu().numpy().astype(np.uint8)

  # Get the outermost contours.
  contours, _ = cv.findContours(raster_flattened_np_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

  # Get rasters with all holes filled by filling the contours.
  filled_flattened_np = np.zeros_like(raster_flattened_np_uint8)
  ctr_idx = -1
  thickness = -1
  cv.drawContours(filled_flattened_np, contours, ctr_idx, (1), thickness)

  # Invert the above rasters, that is, flip all 0's and 1's.
  filled_flattened = torch.from_numpy(filled_flattened_np.astype(np.float32)).cuda()
  inverse_filled_flattened = torch.ones_like(filled_flattened) - filled_flattened
  assert(torch.all((inverse_filled_flattened == 0.) | (inverse_filled_flattened == 1.)))

  # Get the inverse soft masks with all values outside of the mesh's contour set to zero.
  inverse_soft_masks = torch.ones_like(soft_masks) - soft_masks
  inverse_filled = torch.unflatten(inverse_filled_flattened, 0, soft_masks.shape[:2])
  inverse_soft_masks_minus_bdry = inverse_soft_masks - inverse_filled
  inverse_soft_masks_minus_bdry_clamped = torch.clamp(inverse_soft_masks_minus_bdry, min=0., max=None)
  assert(torch.all(inverse_soft_masks_minus_bdry_clamped >= 0.))
  assert(torch.all(inverse_soft_masks_minus_bdry_clamped <= 1.))

  # Finally, get the soft masks but with all holes filled.
  filled_soft_masks = soft_masks + inverse_soft_masks_minus_bdry_clamped
  assert(torch.all(filled_soft_masks >= 0.))
  assert(torch.all(filled_soft_masks <= 1.))
  assert(not torch.equal(torch.unique(filled_soft_masks), torch.tensor([0., 1.]).cuda()))

  return filled_soft_masks
