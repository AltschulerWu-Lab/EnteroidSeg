"""
Cell-type specific segmentation pipelines
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = u'Greys_r'
import numpy as np
import os
from scipy import ndimage as ndi
from skimage import color, filters, io, measure, morphology, restoration, segmentation
import warnings

from utils import imfuns, setting

class Segmentor:
  """
  General segmentation class

  Attributes:
    C (dict): segmentation parameters
    im (float ndarray): image to be segmented
    im_smooth (float ndarray): smoothed image
    im_segmented (label ndarray): labeled objects from segmentation
    object_type: type of object being segmented (dna, crypt, edu, goblet, stem)
  """

  def __init__(self, im, object_type=None):
    """
    Loads attributes

    Args:
      im (float ndarray): image
      object_type (str): type of object to be segmented
    """

    self.im = imfuns.check_im(im)

    self.object_type = object_type

    # store image processing output
    self.im_smooth = []
    self.im_segmented = []

    self.get_params()

  def check_path(self, path):
    """Check if path directory exists"""

    prefix = os.path.dirname(path)
    if not os.path.exists(prefix):
      raise ValueError('Output directory does not exist.')

  def denoise_image(self):
    """
    Abstracted version of denoise_bilateral function. Runs function on raw image using given constants
    """
    return restoration.denoise_bilateral(self.im, sigma_color=self.C['BILATERAL_SIGMA_COLOR'], 
      sigma_spatial=self.C['BILATERAL_SIGMA_SPATIAL'], multichannel=False)

  def get_params(self):
    """Get segmentation parameter setting based on object type"""

    self.C = setting.seg_params[self.object_type]

  def label2rgb(self, im_labeled):
    """
    Abstracted version of label2rgb

    Args:
      im_labeled (labeled ndarray): regions to false color

    Returns:
      rbg ndarray: colored regions overlay on image
    """

    return color.label2rgb(im_labeled, image=self.im, bg_label=0)
  
  def preprocess(self):
    """Runs preprocessing steps (e.g. smooth, threshold)"""
    pass

  def plot_results(self, save=False, show=True):
    """
    Plots results of main steps in pipeline

    Args:
      save (bool): if True, saves output
      show (bool): if True, shows output
    """
    pass

  def run(self, plot=False, save=True):
    """
    Runs and saves segmentation pipeline. Optionally, save results of main pipeline steps

    Args:
      plot (bool): if True, saves results of main pipeline steps
      save (bool): if True, saves output of segmentation
    """

    self.preprocess()
    self.segment()

    if save:
      self.save()

    if plot:
      self.plot_results(save=True, show=False)

  def save(self):
    """
    Saves the segmentation (labeled ndarray) image and segmentation overlay (rbg ndarray) image
    """

    outpath_seg = setting.paths['segmentation'].format(object_type=self.object_type)
    outpath_overlay = setting.paths['overlay'].format(object_type=self.object_type)

    self.check_path(outpath_seg)
    self.check_path(outpath_overlay)

    with warnings.catch_warnings():

      # catches warnings (e.g. low contrast image)
      warnings.simplefilter("ignore")

      io.imsave(outpath_seg, np.array(self.im_segmented).astype(np.uint16))
      io.imsave(outpath_overlay, self.label2rgb(self.im_segmented))

  def segment(self):
    """Runs segmentation step of pipeline"""
    pass

  def segment_watershed(self, im, im_thresh, params, compact=True, line=False):
    """
    Segmentation by first detecting cell locations using scale-space Laplacian of Gaussian blob 
    detection. Cell boundaries are determined using watershed

    Args: 
      im (ndarray): raw image
      im_thresh (bool ndarray): thresholded image
      params (dict): segmentation parameters
      compact (bool): use compact parameter for watershed
      line (bool): if True, draw separating lines in output

    Returns:
      labeled ndarray: segmented objects
    """

    blobs = imfuns.find_blobs(im_thresh, params)

    markers = imfuns.blobs_to_markers(im.shape, blobs)

    im_segmented = self.watershed(im, markers, im_thresh, line=line, compact=compact)

    return markers, im_segmented

  def thresh_otsu(self, im):
    """
    Otsu thresholding modified by a factor (THRESHOLD_FACTOR). Also if the image is blank, the 
    'threshold' is greater than the image max intensity

    Returns:
      float: adjusted Otsu threshold
    """
    try: 
      otsu_thresh = filters.threshold_otsu(im)
      modified_otsu = self.C['THRESHOLD_FACTOR']*otsu_thresh
    except ValueError:
      modified_otsu = np.max(im) + 1

    return modified_otsu

  def watershed(self, im, markers, im_thresh, compact=True, line=False):
    """
    Slightly more abstracted watershed function call

    Args:
      im (ndarray): raw image
      markers (labeled ndarray): labeled seeds 
      im_thresh (ndarray): is 0 at not-cell pixels
      compact (bool): if True, use given constant. Else, use 0
      line (bool): if True, draw separating lines in output

    Returns:
      labeled ndarray: segmented image
    """

    if compact:
      compactness = self.C['WATERSHED_COMPACTNESS']
    else:
      compactness = 0

    im_inverted = ((1-im)*setting.max_px_val).astype(int)

    im_watershed = segmentation.watershed(im_inverted, markers, compactness=compactness, 
      connectivity=self.C['WATERSHED_CONN'], mask=im_thresh!=0, watershed_line=line)

    return imfuns.remove_small_objects(im_watershed, self.C['WATERSHED_MIN_SZ'])

class Crypt_Finder(Segmentor):
  """
  Crypt Segmentation

  Attributes:
    im_dna (ndarray): dna stain image
    im_thresh (float ndarray): thresholded image
  """

  def __init__(self, im, im_dna=None):
    Segmentor.__init__(self, im, object_type='crypt')

    self.im_dna = imfuns.check_im(im_dna)

    # storing images
    self.im_thresh = []

  def preprocess(self):
    self.threshold()

  def segment(self):
    self.im_segmented = measure.label(self.im_threshed>0)

  def threshold(self):
    """Threshold by first removing nuclear stain bleed through"""

    im_subtracted = imfuns.subtract(self.im, self.C['DNA_FACTOR']*self.im_dna)
    im_mask = imfuns.imthresh(im_subtracted, self.C['THRESH']) > 0

    im_closed = morphology.binary_closing(im_mask, selem=morphology.disk(self.C['MORPH_CLOSING_SZ']))
    im_opened = morphology.binary_opening(im_closed, selem=morphology.disk(self.C['MORPH_OPENING_SZ']))
    self.im_threshed = imfuns.remove_small_objects(im_opened, self.C['MIN_SZ']) 
    self.im_threshed = imfuns.mask_im(self.im, self.im_threshed)

  def plot_results(self, save=False, show=True):
    """
    Plots results of main steps in pipeline

    Args:
      save (bool): if True, saves output
      show (bool): if True, shows output
    """

    fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(self.im)
    ax[0].set_title('Raw Image')
    ax[1].imshow(self.im_threshed)
    ax[1].set_title('Thresholded Image')
    ax[2].imshow(self.label2rgb(self.im_segmented))
    ax[2].set_title('Segmentation')

    plt.suptitle('{object_type:s} segmentation'.format(object_type=self.object_type))

    if save:
      outpath = setting.paths['result'].format(object_type=self.object_type)
      fig.savefig(outpath)
      plt.close()
      
    if show:
      plt.show()

class Goblet_Segmentor(Segmentor):
  """
  Goblet Segmentation

  Attributes:
    im_thresh (float ndarray): threshed image where under threshold has value 0, over threshold has 
      original value 
  """

  def __init__(self, im):
    """
    See superclass Segmentor
    """
    Segmentor.__init__(self, im, object_type='goblet')

    # storing images
    self.im_thresh = []

  def preprocess(self):
    self.smooth()
    self.threshold()

  def segment(self):
    markers, self.im_segmented = self.segment_watershed(self.im, self.im_thresh, self.C['LOG_BLOB'])

  def smooth(self):
    """
    Smooths image using median filtering
    """

    self.im_smooth = imfuns.filter_median(self.im, self.C['MEDIAN_FILTER_SZ'])

  def threshold(self):
    """
    Thresholds by first finding the Otsu threshold. Holes are removed from Otsu thresholded image and
    convex hulls are created for foreground objects. Foreground objects are expanded to fill the convex
    hulls.
    """
    
    thresh_val = self.thresh_otsu(self.im_smooth)
    
    # convex hull threshold result
    thresh_mask = self.im_smooth > thresh_val
    thresh_mask = ndi.binary_fill_holes(thresh_mask)
    thresh_mask = morphology.convex_hull_object(thresh_mask)

    self.im_thresh = self.im_smooth
    self.im_thresh[~thresh_mask] = 0

    return thresh_val

  def plot_results(self, save=False, show=True):
    """
    Plots results of main steps in pipeline

    Args:
      save (bool): if True, saves output
      show (bool): if True, shows output
    """

    fig, axes = plt.subplots(1, 3, figsize=(21,7), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(self.im)
    ax[0].set_title('Raw Image')
    ax[1].imshow(self.im_thresh)
    ax[1].set_title('Smoothed & Thresholded Image')
    ax[2].imshow(self.label2rgb(self.im_segmented))
    ax[2].set_title('Goblet Segmentation')

    plt.suptitle('goblet segmentation')

    if save:
      outpath = setting.paths['result'].format(object_type=self.object_type)
      fig.savefig(outpath)
      plt.close()
      
    if show:
      plt.show()

class Nuclear_Segmentor(Segmentor):
  """
  Nuclear Segmentation

  Attributes:
    im_thresh (float ndarray): threshed image where under threshold has value 0, over threshold has 
      original value 
    im_clumps (float ndarray): image of nuclear stain showing only clump regions
    seg_firstpass (labeled ndarray): first pass segmentation prior to filtering out clumps
    seg_dense (labeled ndarray): dense segmentation result
    seg_sparse (labeled ndarray): sparse segmentation result
  """

  def __init__(self, im):
    """
    See superclass Segmentor
    """
    Segmentor.__init__(self, im, object_type='dna')

    # storing images
    self.im_thresh = []
    self.seg_firstpass = []
    self.seg_sparse = []
    self.im_clumps = []
    self.seg_dense = []

  def find_clumps(self):
    """
    Find clumps in segmentation image. Clumps include: all objects larger than SEG_SINGLE_MAX_SZ and
    objects between SEG_SINGLE_MIN_SZ and SEG_SINGLE_MAX_SZ that are irregular. Creates a raw image 
    masked to show only clumped regions. Filters clumps from sparse segmentation
    """

    seg_large_clumps = imfuns.remove_small_objects(self.seg_firstpass, self.C['SEG_SINGLE_MAX_SZ'])
    
    seg_mixed = imfuns.remove_small_objects(self.seg_firstpass, self.C['SEG_SINGLE_MIN_SZ'])
    seg_mixed[seg_large_clumps!=0]=0
    seg_mixed_irregular = self.find_irregular_objects(seg_mixed, self.C['SEG_CLUMP_SOLIDITY'])

    seg_clumps = np.maximum(seg_mixed_irregular, seg_large_clumps) 
    seg_clumps = imfuns.remove_small_holes(seg_clumps, self.C['SEG_CLOSE_HOLES'])

    self.im_clumps = np.copy(self.im)
    self.im_clumps[~seg_clumps]=0

    self.seg_sparse = np.copy(self.seg_firstpass)
    self.seg_sparse[self.im_clumps!=0] = 0

  def find_irregular_objects(self, im, solidity_thresh):
    """
    Identifies irregular objects in image

    Args: 
      im (labeled ndarray): image of objects
      solidity_thresh (float): cutoff of solidity value below which an object is irregular

    Returns:
      labeled ndarray: Containing only irregular objects in image
    """
    irregular_labels = [x.label for x in measure.regionprops(im) if x.solidity < solidity_thresh]
    im_irregular = np.copy(im)
    im_irregular[~np.isin(im, irregular_labels)] = 0

    return im_irregular

  def preprocess(self):
    self.threshold()

  def segment(self):
    """
    Runs first sparse segmentation, then dense segmentation. Final output combines the two results
    """

    self.segment_sparse()

    self.find_clumps()

    self.segment_dense()

    self.segment_combine()

  def segment_combine(self):
    """
    Combine sparse and dense seg
    """
    label_add = np.max(self.seg_sparse)+1
    seg_dense_relabeled = self.seg_dense + label_add
    seg_dense_relabeled[seg_dense_relabeled == label_add] = 0

    self.im_segmented = np.maximum(seg_dense_relabeled, self.seg_sparse)

  def segment_dense(self):
    """
    Performs dense segmentation by segmenting the clumps using dense seg parameters
    """
    markers, self.seg_dense = self.segment_watershed(self.im_clumps, self.im_clumps, 
      self.C['LOG_DENSE'], line=True)    

  def segment_sparse(self):
    """
    Performs sparse segmentation
    """

    markers, self.seg_firstpass = self.segment_watershed(self.im, self.im_thresh, 
      self.C['LOG_SPARSE'], compact=False)

  def threshold(self):
    """
    Threshold using Otsu
    """

    self.im_smooth = self.denoise_image()
    thresh_val = self.thresh_otsu(self.im_smooth)

    self.im_thresh = np.copy(self.im_smooth)
    self.im_thresh[self.im_smooth < thresh_val] = 0

    return thresh_val

  def plot_results(self, save=False, show=True):
    """
    Plots results of main steps in pipeline

    Args:
      save (bool): if True, saves output
      show (bool): if True, shows output
    """

    fig, axes = plt.subplots(2, 4, figsize=(28,14), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title('Max projected image')
    ax[0].imshow(self.im)

    ax[1].set_title('Sparse segmentation')
    ax[1].imshow(self.label2rgb(self.seg_sparse))

    ax[4].set_title('Thresholded image')
    ax[4].imshow(self.im_thresh)

    ax[5].set_title('Dense segmentation')
    ax[5].imshow(self.label2rgb(self.seg_dense))

    plt.subplot(1,2,2, sharex=ax[0], sharey=ax[0])
    plt.title('Final Segmentation')
    plt.imshow(self.label2rgb(self.im_segmented))

    plt.suptitle('{object_type:s} segmentation'.format(object_type=self.object_type))

    if save:
      outpath = setting.paths['result'].format(object_type=self.object_type)
      fig.savefig(outpath)
      plt.close()
      
    if show:
      plt.show()

class EdU_Segmentor(Nuclear_Segmentor):
  """
  See superclass Nuclear_Segmentor
  """

  def __init__(self, im):
    Nuclear_Segmentor.__init__(self, im)

    self.object_type = 'edu'
    self.get_params()

class Stem_Segmentor(Crypt_Finder):
  """
  Stem cell segmentation

  Attributes:
    crypt_mask (bool ndarray): value 1 if crypt region, 0 otherwise
  """

  def __init__(self, im, im_dna, objects_paneth=None):
    """
    See superclass Crypt_Finder
    """
    Crypt_Finder.__init__(self, im)

    self.object_type = 'stem'
    self.get_params()

    self.im_dna = imfuns.check_im(im_dna)
    self.objects_paneth = objects_paneth

    self.objects_crypt = self.segment_crypt()
    self.objects_dna = self.segment_dna()

    # self.objects_paneth = objects_paneth

  def segment_crypt(self):
    """Segment required input objects"""
    crypt_seg = Crypt_Finder(self.im, im_dna=self.im_dna)
    crypt_seg.run(save=False)

    return crypt_seg.im_segmented

  def segment_dna(self):
    """Segment required input objects"""

    nuclei_seg = Nuclear_Segmentor(self.im_dna)
    nuclei_seg.run(save=False)

    return nuclei_seg.im_segmented

  def filter_paneth(self):
    """
    Filter out Paneth nuclei (assigned as nuclei closest to centroid of Paneth objects)
    """

    paneth_labels = imfuns.assign_centroids(self.im_segmented, self.objects_paneth)

    self.im_segmented = imfuns.remove_regions(self.im_segmented, paneth_labels)


  def filter_partial(self):
    """
    Filter out nuclei partially in the crypt. Partial nuclei are defined as nuclei
    where ratio of the area outside the crypt to the area inside the crypt > PARTIAL_RATIO
    """
 
    self.im_segmented = imfuns.overlap_regions(self.objects_dna, 
      self.objects_crypt, self.C['PARTIAL_RATIO'])

  def preprocess(self):
    pass

  def segment(self):
    """
    Identify stem nuclei in crypts (filter out Paneth and partial nuclei)
    """

    self.filter_partial()

    if self.objects_paneth is not None:
      self.filter_paneth()

  def plot_results(self, save=False, show=True):
    """
    Plots results of main steps in pipeline

    Args:
      save (bool): if True, saves output
      show (bool): if True, shows output
    """

    fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(self.im)
    ax[0].set_title('Raw Image')
    ax[1].imshow(self.label2rgb(self.objects_crypt))
    ax[1].set_title('Crypt Objects')
    ax[2].imshow(self.label2rgb(self.objects_dna))
    ax[2].set_title('Nuclear Objects')
    ax[3].imshow(self.label2rgb(self.im_segmented))
    ax[3].set_title('Segmentation')

    plt.suptitle('{object_type:s} segmentation'.format(object_type=self.object_type))

    if save:
      outpath = setting.paths['result'].format(object_type=self.object_type)
      fig.savefig(outpath)
      plt.close()
      
    if show:
      plt.show()




    
