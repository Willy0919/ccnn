#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: See LICENSE.txt
    @contact: daniel.onoro@edu.uah.es
    @date: 27/02/2015
'''

"""
Test script. This code performs a test over with a pre trained model over the
specified dataset.
"""

#===========================================================================
# Dependency loading
#===========================================================================
# File storage
import h5py
import scipy.io as sio

# System
import signal
import sys, getopt
signal.signal(signal.SIGINT, signal.SIG_DFL)
import time
#sys.path.append("..")
# Vision and maths
import numpy as np
import utils as utl
from gen_features import genDensity, genPDensity, loadImage, extractEscales
import caffe
import cv2


class CaffePredictor:
    def __init__(self, prototxt, caffemodel, n_scales):
        # Load a precomputed caffe model
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data_s0'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))  # It's already RGB
        # Reshape net for the single input
        b_shape = self.net.blobs['data_s0'].data.shape
        self._n_scales = n_scales
        for s in range(n_scales):
            scale_name = 'data_s{}'.format(s)
            self.net.blobs[scale_name].reshape(b_shape[0], b_shape[1], b_shape[2], b_shape[3])

    # Probably it is not the eficient way to do it...
    def process(self, im, base_pw):
        # Compute dense positions where to extract patches
        [heith, width] = im.shape[0:2]
        pos = utl.get_dense_pos(heith, width, base_pw, stride=10)

        # Initialize density matrix and vouting count
        dens_map = np.zeros((heith, width), dtype=np.float32)  # Init density to 0
        count_map = np.zeros((heith, width), dtype=np.int32)  # Number of votes to divide

        # Iterate for all patches
        for ix, p in enumerate(pos):
            # Compute displacement from centers
            dx = dy = int(base_pw / 2)

            # Get roi
            x, y = p
            sx = slice(x - dx, x + dx + 1, None)
            sy = slice(y - dy, y + dy + 1, None)
            crop_im = im[sx, sy, ...]
            h, w = crop_im.shape[0:2]
            if h != w or (h <= 0):
                continue

            # Get all the scaled images
            im_scales = extractEscales([crop_im], self._n_scales)

            # Load and forward CNN
            for s in range(self._n_scales):
                data_name = 'data_s{}'.format(s)
                self.net.blobs[data_name].data[...] = self.transformer.preprocess('data', im_scales[0][s].copy())
            self.net.forward()

            # Take the output from the last layer
            # Access to the last layer of the net, second element of the tuple (layer, caffe obj)
            pred = self.net.blobs.items()[-1][1].data

            # Make it squared
            p_side = int(np.sqrt(len(pred.flatten())))
            pred = pred.reshape((p_side, p_side))

            # Resize it back to the original size
            pred = utl.resizeDensityPatch(pred, crop_im.shape[0:2])
            pred[pred < 0] = 0

            # Sumup density map into density map and increase count of votes
            dens_map[sx, sy] += pred
            count_map[sx, sy] += 1

        # Remove Zeros
        count_map[count_map == 0] = 1

        # Average density map
        dens_map = dens_map / count_map

        return dens_map
#===========================================================================
# Some helpers functions
#===========================================================================
def testOnImg(CNN, im, pw, mask = None):
    
    # Process Image
    resImg = CNN.process(im, pw) 
    realmap = resImg
    # Mask image if provided
    if mask is not None:
        resImg = resImg * mask
        #gtdots = gtdots * mask

    npred=resImg.sum()
    #ntrue=gtdots.sum()

    return npred,resImg,realmap#,gtdots

def initTestFromCfg(cfg_file):
    '''
    @brief: initialize all parameter from the cfg file. 
    '''
    
    # Load cfg parameter from yaml file
    cfg = utl.cfgFromFile(cfg_file)
    
    # Fist load the dataset name
    dataset = cfg.DATASET
    
    # Set default values
    use_mask = cfg[dataset].USE_MASK
    use_perspective = cfg[dataset].USE_PERSPECTIVE
    
    # Mask pattern ending
    mask_file = cfg[dataset].MASK_FILE
        
    # Img patterns ending
    dot_ending = cfg[dataset].DOT_ENDING
    
    # Test vars
    test_names_file = cfg[dataset].TEST_LIST
    
    # Im folder
    im_folder = cfg[dataset].IM_FOLDER
    
    # Results output foder
    results_file = cfg[dataset].RESULTS_OUTPUT

    # Resize image
    resize_im = cfg[dataset].RESIZE

    # Patch parameters
    pw = cfg[dataset].PW # Patch with 
    sigmadots = cfg[dataset].SIG # Densities sigma
    n_scales = cfg[dataset].N_SCALES # Escales to extract
    perspective_path = cfg[dataset].PERSPECTIVE_MAP
    is_colored = cfg[dataset].COLOR

        
    return (dataset, use_mask, mask_file, test_names_file, im_folder, 
            dot_ending, pw, sigmadots, n_scales, perspective_path, 
            use_perspective, is_colored, results_file, resize_im)


def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--cpu_only"
    print "\t--tdev <GPU ID>"
    print "\t--prototxt <caffe prototxt file>"
    print "\t--caffemodel <caffe caffemodel file>"
    print "\t--cfg <config file yaml>"

def main(argv):
    # Init parameters      
    use_cpu = True
    gpu_dev = 0

    # Batch size
    b_size = -1

    # CNN vars
    prototxt_path = 'models/ucsd/hydra3/hydra3_deploy.prototxt'
    caffemodel_path = 'models/ucsd/hydra3/trancos_hydra2.caffemodel'
        
        
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["prototxt=", "caffemodel=", 
                                             "cpu_only", "dev=", "cfg=", "img="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--prototxt"):
            prototxt_path = arg
        elif opt in ("--caffemodel"):
            caffemodel_path = arg
        elif opt in ("--cpu_only"):
            use_cpu = True            
        elif opt in ("--dev"):
            gpu_dev = int(arg)
        elif opt in ("--cfg"):
            cfg_file = arg
        elif opt in ("--img"):
            image_file = arg
            
    print "Loading configuration file: ", cfg_file
    (dataset, use_mask, mask_file, test_names_file, im_folder, 
            dot_ending, pw, sigmadots, n_scales, perspective_path, 
            use_perspective, is_colored, results_file, resize_im) = initTestFromCfg(cfg_file)
            
    print "Choosen parameters:"
    print "-------------------"
    print "Use only CPU: ", use_cpu
    print "GPU devide: ", gpu_dev
    print "Dataset: ", dataset
    print "Results files: ", results_file
    print "Test data base location: ", im_folder
    print "Test inmage names: ", test_names_file
    print "Dot image ending: ", dot_ending
    print "Use mask: ", use_mask
    print "Mask pattern: ", mask_file
    print "Patch width (pw): ", pw
    print "Sigma for each dot: ", sigmadots
    print "Number of scales: ", n_scales
    print "Perspective map: ", perspective_path
    print "Use perspective:", use_perspective
    print "Prototxt path: ", prototxt_path
    print "Caffemodel path: ", caffemodel_path
    print "Batch size: ", b_size
    print "Resize images: ", resize_im
    print "==================="
    
    print "----------------------"
    print "Preparing for Testing"
    print "======================"

    # Set GPU CPU setting
    if use_cpu:
        caffe.set_mode_cpu()
    else:
        # Use GPU
        caffe.set_device(gpu_dev)
        caffe.set_mode_gpu()

    print "Reading perspective file"
    if use_perspective:
        pers_file = h5py.File(perspective_path,'r')
        pers_file.close()
        
    mask = None
    if dataset == 'UCSD':
        print "Reading mask"
        if use_mask:
            mask_f = h5py.File(mask_file,'r')
            mask = np.array(mask_f['mask'])
            mask_f.close()
    
    # Init CNN
    CNN = CaffePredictor(prototxt_path, caffemodel_path, n_scales)

    # Read image files
    im = loadImage(image_file, color = is_colored)

    if resize_im > 0:
        # Resize image
        im = utl.resizeMaxSize(im, resize_im)

    s=time.time()
    npred,resImg,realImg=testOnImg(CNN, im, pw, mask)

    print "npred = %.2f , time =%.2f sec"%(npred,time.time()-s)
    sio.savemat('predictionmap.mat', {'d_map':realImg})

    
    return 0

if __name__=="__main__":
    main(sys.argv[1:])
