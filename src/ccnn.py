__author__ = 'willy'

import h5py
import scipy.io as sio

# System
import signal
import sys, getopt
signal.signal(signal.SIGINT, signal.SIG_DFL)
import time
#sys.path.append("..")
#print '======',sys.path
# Vision and maths
import numpy as np
import utils as utl
from gen_features import genDensity, genPDensity, loadImage, extractEscales
import caffe
import cv2
from demo import CaffePredictor,testOnImg,initTestFromCfg,dispHelp
import os

class CCNN(object):
    def __init__(self, use_cpu=False, gpu_dev=0):
	#print '????????????',__file__
	parent_path = os.path.dirname(os.path.dirname(__file__))
	print '?????',parent_path
        self.use_cpu = use_cpu
        self.gpu_dev = gpu_dev
	#CCNN
        self._conf = os.path.join(parent_path, 'models/ucsd/ccnn/ccnn_ucsd_cfg.yml')
        self._caffe_model = os.path.join(parent_path, 'models/pretrained_models/ucsd/ccnn/ucsd_ccnn_up.caffemodel')
        self._prototxt = os.path.join(parent_path, 'models/ucsd/ccnn/ccnn_deploy.prototxt')
	#HYDRA3
	#self._conf = 'models/ucsd/hydra3/hydra3_ucsd_cfg.yml'
        #self._caffe_model = 'models/pretrained_models/ucsd/hydra3/ucsd_hydra3_up.caffemodel'
        #self._prototxt = 'models/ucsd/hydra3/hydra3_deploy.prototxt'
        # Batch size
        self.b_size = -1
        (self.dataset, self.use_mask, self.mask_file, self.test_names_file, self.im_folder,
         self.dot_ending, self.pw, self.sigmadots, self.n_scales, self.perspective_path,
         self.use_perspective, self.is_colored, self.results_file, self.resize_im) = initTestFromCfg(self._conf)
        print "Choosen parameters:"
        print "-------------------"
        print "Use only CPU: ", self.use_cpu
        print "GPU devide: ", self.gpu_dev
        print "Dataset: ", self.dataset
        print "Results files: ", self.results_file
        print "Test data base location: ", self.im_folder
        print "Test inmage names: ", self.test_names_file
        print "Dot image ending: ", self.dot_ending
        print "Use mask: ", self.use_mask
        print "Mask pattern: ", self.mask_file
        print "Patch width (pw): ", self.pw
        print "Sigma for each dot: ", self.sigmadots
        print "Number of scales: ", self.n_scales
        print "Perspective map: ", self.perspective_path
        print "Use perspective:", self.use_perspective
        print "Prototxt path: ", self._prototxt
        print "Caffemodel path: ", self._caffe_model
        print "Batch size: ", self.b_size
        print "Resize images: ", self.resize_im
        print "==================="

        print "----------------------"
        print "Preparing for Testing"
        print "======================"

        if self.use_cpu:
	    print '+++++++'
            caffe.set_mode_cpu()
        else:
            # Use GPU
            caffe.set_device(self.gpu_dev)
            caffe.set_mode_gpu()


        self.mask = None
        if self.dataset == 'UCSD':
            print "Reading mask"
            if self.use_mask:
                mask_f = h5py.File(os.path.join(parent_path,self.mask_file), 'r')
                self.mask = np.array(mask_f['mask'])
                mask_f.close()

        # Init CNN
        self.CNN = CaffePredictor(os.path.join(parent_path,self._prototxt), os.path.join(parent_path,self._caffe_model), self.n_scales)

    def getCrowdCount(self,img_path):
        # Read image files
        im = loadImage(img_path, color=self.is_colored)

        if self.resize_im > 0:
            # Resize image
            im = utl.resizeMaxSize(im, self.resize_im)

        s = time.time()
#	print 'pw:',self.pw
        npred,resImg,realmap = testOnImg(self.CNN, im, self.pw, self.mask)
        return npred,time.time() - s


if __name__ == '__main__':

    ccnn = CCNN(use_cpu=True)
    image_path = [os.path.join(parent_path,'projects/pics1.jpg'),os.path.join(parent_path,'projects/pics19.jpg')]
    for image in image_path:
	print image
        (count, t) = ccnn.getCrowdCount(image)
    	print 'count = %.2f, time: %.2f s'%(count, t)
