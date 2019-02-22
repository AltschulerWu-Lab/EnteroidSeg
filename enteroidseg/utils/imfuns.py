"""
Helpers image processing functions
==================================

"""
import math
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology
import warnings

from utils import setting

def assign_centroids(im_labeled, target_objects):
  """
  Assigned objects in im_labeled closest to centroids of objects in target_objects)

  Args:
    im_labeled (label ndarray): segmentation image to assign to centroids
    target_objects (label ndarray): segmentation image to determine centroid locations

  Returns:
    label ndarray: objects in im_label closest to centroids
  """

  centroids = [np.round(el.centroid).astype(int) for el in measure.regionprops(target_objects)]

  if not centroids:
    return [] 

  x, y = zip(*centroids)
  z = np.ones(len(x));

  target_seeds = np.zeros(target_objects.shape).astype(int)
  target_seeds[x,y] = z

  target_seeds_labeled = mask_im(im_labeled, target_seeds)
  target_labels = nonzero_list(target_seeds_labeled)

  return target_labels

def blobs_to_markers(im_shape, blobs):
  """
  Converts output of blob detection to matrix containing center of blobs (seeds)  

  Args:
    im_shape (tuple): dimension of image (e.g. (2,3) for 2 by 3 pixel image)
    blobs (list of tuples): list of (x, y, ~, ...) indicating x, y position of 
      blob centers

  Returns:
    labeled ndarray: image with labeled center positions (size as provided)

  """

  im_seeds = np.zeros(im_shape)
  im_seeds[blobs[:,0].astype(int), blobs[:,1].astype(int)] = 1

  markers, markers_num = ndi.label(im_seeds)

  return markers

def check_im(im):
  """Convert image to float if it is not a float array """

  if im is None:
    return None

  if im.dtype != np.dtype('float'):
    im = im.astype(float)/setting.max_px_val

  return im

def filter_median(im, filter_size):
  """
  Abstracted media filter function that catches warnings about image type conversion (float to uint8)

  Args:
    im (ndarray): image
    filter_size (int): size of disk filter

  Returns:
    uint8 ndarray: filtered image
  """

  with warnings.catch_warnings():

    # catches warnings (e.g. about labeled)
    warnings.simplefilter("ignore")

    return filters.median(im, selem=morphology.disk(filter_size))

def find_blobs(im, p):
  """
  Abstracted version of blob_log function

  Args: 
    im (ndarray): input image
    p (dict): parameters for function (must include keys 'MIN_SIG', 'MAX_SIG', 
      'NUM_SIG', 'THRESH', and 'OVERLAP')

  Returns:
    list of tuples: [(x, y, r)] where x and y are coordinates of blob center and 
      r is radius of blob
  """

  blobs = feature.blob_log(im, min_sigma=p['MIN_SIG'], max_sigma=p['MAX_SIG'], 
    num_sigma=p['NUM_SIG'], threshold=p['THRESH'], overlap=p['OVERLAP'])

  blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

  return blobs

def keep_regions(im, labels, bg_val=0):
  """
  Keep regions with label in given list from segmentation image
  
  Args:
    im (label ndarray): segmentation image
    labels (int list): list of labels of objects to keep
    bg_val (int): value of background region
  
  Returns:
    label ndarray: segmentation image with only objects in labels list
  """

  new_im = im.copy()
  new_im[~np.isin(new_im, labels)] = bg_val

  return new_im

def imthresh(im, thresh):
  """
  Sets pixels in image below threshold value to 0

  Args:
    im (ndarray): image
    thresh (float): threshold

  Returns: 
    ndarray: thresholded image
  """
  
  thresh_im = im.copy()
  thresh_im[thresh_im < thresh] = 0
  return thresh_im

def mask_im(im, mask, val=0):
  """
  Sets pixels not in mask to 0 (or val) in image
  
  Args:
    im (ndarray): image
    mask (bool ndarray): mask where 1 indicates regions to keep
    val (int): value to set regions outside of mask
  
  Returns:
    ndarray: masked image
  """

  masked_im = im.copy() 
  masked_im[mask == 0] = val
  return masked_im

def nonzero_list(im):
  """Returns image pixels as a list of unique labels without the 0s"""

  uniques = list(np.unique(im))
  nonzero = [el for el in uniques if el != 0]
  return nonzero

def overlap_regions(im, mask, partial_ratio):
  """
  Filter out objects partially in the mask. Partial objects are defined as objects
  where ratio of the area outside the mask to the area inside the mask > partial_ratio

  Args:
    im (label ndarray): segmentation image
    mask (bool ndarray): mask where 1 indicates regions to keep
    partial_ratio (float): ratio of area outside the mask to area inside the mask

  Returns:
    labeled ndarray: containing objects considered in the mask
  """

  im_in = mask_im(im, mask)

  in_rp = measure.regionprops(im_in)
  in_areas = [el.area for el in in_rp]
  in_labels = [el.label for el in in_rp]

  im_potential = keep_regions(im, in_labels)
  im_out = mask_im(im_potential, ~(mask>0))

  out_rp = measure.regionprops(im_out)
  out_areas = [el.area for el in out_rp]
  out_labels = [el.label for el in out_rp]

  remove_labels = []

  for out_idx, label in enumerate(out_labels):
    out_area = out_areas[out_idx]

    in_idx = in_labels.index(label)
    in_area = in_areas[in_idx]

    if out_area/in_area > partial_ratio:
      remove_labels.append(label)

  im_filtered = remove_regions(im_potential, remove_labels)

  return im_filtered

def remove_regions(im, labels, bg_val=0):
  """
  Remove regions with label in given list from segmentation image
  
  Args:
    im (label ndarray): segmentation image
    labels (int list): list of labels of objects to be removed
    bg_val (int): value of background region

  Returns:
    ndarray: segmentation image without objects in labels list
  """

  new_im = im.copy()
  new_im[np.isin(new_im, labels)] = bg_val

  return new_im

def remove_small_holes(im, min_size):
  """
  Abstracted remove_small_holes function that catches the warnings about labeled arrays

  Args:
    im (labeled/bool ndarray): image with holes
    min_size (int): size below which to remove holes

  Returns:
    bool ndarray: labeled ndarray inputs will be converted into bool ndarrays
  """

  with warnings.catch_warnings():

    # catches warnings (e.g. about labeled)
    warnings.simplefilter("ignore")
    
    return morphology.remove_small_holes(im, min_size=min_size)

def remove_small_objects(im, min_size):
  """
  Abstracted remove_small_objects function that catches the warnings about labeled arrays

  Args:
    im (labeled/bool ndarray): image with holes
    min_size (int): size below which to remove holes

  Returns:
    bool ndarray: labeled ndarray inputs will be converted into bool ndarrays
  """

  with warnings.catch_warnings():

    # catches warnings (e.g. about labeled)
    warnings.simplefilter("ignore")
    
    return morphology.remove_small_objects(im, min_size=min_size)

def subtract(im1, im2):
  """Subtract two images (im1-im2). Pixel value cannot go below 0"""

  im = im1 - im2
  im[im<0] = 0

  return im


