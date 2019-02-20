import segmentation
from skimage import io
from utils import setting

im_path = 'images/im_edu.tiff'

im = io.imread(im_path)
s = segmentation.EdU_Segmentor(im)
s.run(plot=True)