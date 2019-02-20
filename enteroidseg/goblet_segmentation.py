import segmentation
from skimage import io
from utils import setting

im_path = 'images/im_muc2.tiff'

im = io.imread(im_path)
s = segmentation.Goblet_Segmentor(im)
s.run(plot=True)