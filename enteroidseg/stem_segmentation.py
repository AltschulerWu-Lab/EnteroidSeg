import segmentation
from skimage import io

# Stem Segmentation

# Setup segmentation
im_path = 'images/im_lgr5.tiff'
im_dna_path = 'images/im_hoechst.tiff'
objects_paneth_path = 'images/objects_paneth.tiff'

im = io.imread(im_path)
im_dna = io.imread(im_dna_path)
objects_paneth = io.imread(objects_paneth_path)

s = segmentation.Stem_Segmentor(im, im_dna, objects_paneth=objects_paneth)

# Set plot=True if you want to plot and save the results of main pipeline steps
s.run(plot=True)