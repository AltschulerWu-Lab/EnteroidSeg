import segmentation
from skimage import io

# EdU Segmentation

# Setup segmentation
im_path = 'images/im_edu.tiff'
im = io.imread(im_path)

s = segmentation.EdU_Segmentor(im)

# Set plot=True if you want to plot and save the results of main pipeline steps
s.run(plot=True)