__author__ = 'willy'

import sys, getopt

# File storage
import h5py

# Vision and maths
import numpy as np
import utils as utl
import skimage.io
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import scipy.io as sio


def genPDensity(dot_im, sigmadots, pmap):

    # Initialize density map
    dmap = np.zeros((dot_im.shape[0], dot_im.shape[1]), np.float32)

    # Get notation positions
    pos_list = getGtPos(dot_im)
    for pos in pos_list:
        x, y = pos
        g = 1 / pmap[x, y]

        h = np.zeros_like(dmap)
        h[x, y] = 1.0
        h = gaussian_filter(h, sigmadots * g)

        dmap = dmap + h

    return dmap


def getGtPos(dot_im):
    '''
    @brief: This function gets a dotted image and returns the ground truth positions.
    @param dots: annotated dot image.
    @return: matrix with the notated object positions.
    '''
    dot = dot_im[:, :, 0] / dot_im[:, :, 0].max()

    # Find positions
    pos = np.where(dot == 1)

    pos = np.asarray((pos[0], pos[1])).T

    return pos

def loadImage(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Take
    filename: string
    color: flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).
    Give
    image: an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


if __name__ == "__main__":
    dot_image = r'/home/wl/caffe/examples/ccnn/data/UCSD/images/vidf1_33_003_f004dots.png'
    perspective_path = r'/home/wl/caffe/examples/ccnn/data/UCSD/params/ucsd_pmap_min_norm.h5'
    #perspective map
    pers_file = h5py.File(perspective_path, 'r')
    pmap = np.array(pers_file['pmap'])
    pers_file.close()
    # load dot image
    dot_im = loadImage(dot_image, color = True)
    dens_im = genPDensity(dot_im, 8.0, pmap)
    sio.savemat('gt_map.mat', {'d_map': dens_im})
