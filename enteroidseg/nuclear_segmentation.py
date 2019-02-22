import segmentation
from skimage import io

# Nuclei Segmentation

# Setup segmentation
im_path = 'images/im_hoechst.tiff'
im = io.imread(im_path)

s = segmentation.Nuclear_Segmentor(im)

# Set plot=True if you want to plot and save the results of main pipeline steps
s.run(plot=True)
