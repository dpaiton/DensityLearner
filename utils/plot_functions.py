import os
import numpy as np
import matplotlib.pyplot as plt
import utils.image_processing as ip

"""
Save figure for input data as a tiled image
Inpus:
  data: [np.ndarray] of shape:
    (height, width) - single image
    (n, height, width) - n images tiled with dim (height, width)
    (n, height, width, features) - n images tiled with dim
      (height, width, features); features could be color
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
    figures will not display with GUI if set
  vmin, vmax: [int] the min and max of the color range
"""
def save_data_tiled(data, normalize=False, title="", save_filename="",
  vmin=None, vmax=None):
  if normalize:
    data = ip.normalize_data_with_max(data)
    vmin = -1.0
    vmax = 1.0
  if vmin is None:
    vmin = np.min(data)
  if vmax is None:
    vmax = np.max(data)
  if len(data.shape) >= 3:
    data = pad_data(data)
  fig, sub_axis = plt.subplots(1)
  axis_image = sub_axis.imshow(data, cmap="Greys_r", interpolation="nearest")
  axis_image.set_clim(vmin=vmin, vmax=vmax)
  cbar = fig.colorbar(axis_image)
  sub_axis.tick_params(
   axis="both",
   bottom="off",
   top="off",
   left="off",
   right="off")
  sub_axis.get_xaxis().set_visible(False)
  sub_axis.get_yaxis().set_visible(False)
  sub_axis.set_title(title)
  if save_filename == "":
    save_filename = "./output.ps"
  fig.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
  plt.close(fig)

"""
Pad data with ones for visualization
Outputs:
  padded version of input
Inputs:
  data: np.ndarray
"""
def pad_data(data):
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1)) # add some space between filters
    + ((0, 0),) * (data.ndim - 3)) # don't pad last dimension (if there is one)
  padded_data = np.pad(data, padding, mode="constant", constant_values=1)
  # tile the filters into an image
  padded_data = padded_data.reshape((
    (n, n) + padded_data.shape[1:])).transpose((
    (0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1))))
  padded_data = padded_data.reshape((n * padded_data.shape[1],
    n * padded_data.shape[3]) + padded_data.shape[4:])
  return padded_data
