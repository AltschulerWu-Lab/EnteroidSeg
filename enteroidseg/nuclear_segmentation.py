import segmentation
from skimage import io
from utils import setting

im_path = 'images/im_hoechst.tiff'

im = io.imread(im_path)
s = segmentation.Nuclear_Segmentor(im)
s.run(plot=True)
