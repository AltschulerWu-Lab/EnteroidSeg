import segmentation
from skimage import io
from utils import setting

im_path = 'images/im_lgr5.tiff'
dna_obj_path = 'images/dna_obj.tiff'
crypt_obj_path = 'images/crypt_obj.tiff'

im = io.imread(im_path)
crypt_obj = io.imread(crypt_obj_path)
dna_obj = io.imread(dna_obj_path)
s = segmentation.Stem_Segmentor(im, crypt_obj, dna_obj)
s.run(plot=True)