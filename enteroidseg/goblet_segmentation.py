import segmentation
from skimage import io

# Goblet Segmentation

# Setup segmentation
im_path = 'images/im_muc2.tiff'
im = io.imread(im_path)

s = segmentation.Goblet_Segmentor(im)

# Set plot=True if you want to plot and save the results of main pipeline steps
s.run(plot=True)