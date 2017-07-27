__author__ = 'willy'

__author__ = 'willy'
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2

CLASSES = ('__background__',
           'person')


#          'aeroplane', 'bicycle', 'bird', 'boat',
#          'bottle', 'bus', 'car', 'cat', 'chair',
#          'cow', 'diningtable', 'dog', 'horse',
#          'motorbike', 'person', 'pottedplant',
#          'sheep', 'sofa', 'train', 'tvmonitor')

class Pedestrain(object):
    def __init__(self, use_cpu=False, gpu_dev=0):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        self.prototxt = os.path.join(cfg.MODELS_DIR, 'VGG16',
                                     'faster_rcnn_end2end', 'test.prototxt')
        self.caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                                       'VGG16_faster_rcnn_office.caffemodel')

        if not os.path.isfile(self.caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(self.caffemodel))

        if use_cpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_dev)
            cfg.GPU_ID = gpu_dev
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

        print '\n\nLoaded network {:s}'.format(self.caffemodel)

    #    def get_count(self, img_path, region):
    #        """Detect object classes in an image using pre-computed object proposals."""
    #	#transform region
    #	points = []
    #	for xy in region:
    #	    points.append([xy['X'], xy['Y']])
    #
    #        # Load the demo image
    #        im_file = os.path.join(cfg.DATA_DIR, 'demo', img_path)
    #        im = cv2.imread(im_file)
    #
    #	#get region
    #	mask = np.zeros(im.shape, np.uint8)
    #	pts = np.array(points, np.int32)
    #	pts = pts.reshape(-1,1,2)
    #	cv2.fillPoly(mask, [pts],(1,1,1))
    #	im = im* mask
    #
    #        # Detect all object classes and regress object bounds
    #        timer = Timer()
    #        timer.tic()
    #        scores, boxes = im_detect(self.net, im)
    #        timer.toc()
    #        print ('Detection took {:.3f}s for '
    #               '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #
    #        # Visualize detections for each class
    #        CONF_THRESH = 0.5
    #        NMS_THRESH = 0.3
    #        for cls_ind, cls in enumerate(CLASSES[1:]):
    #            cls_ind += 1  # because we skipped background
    #            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    #            cls_scores = scores[:, cls_ind]
    #            dets = np.hstack((cls_boxes,
    #                              cls_scores[:, np.newaxis])).astype(np.float32)
    #            keep = nms(dets, NMS_THRESH)
    #            dets = dets[keep, :]
    #	    if cls == "person":
    #                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    #	    #for i in inds:
    #	    #    print "class and num",CLASSES[i]
    #                return len(inds)

    def get_pre(self, im, region):
        # Load the demo image
        points = []
        if len(region) > 0:
            # transform region

            for xy in region:
                points.append([xy['X'], xy['Y']])
        #im = cv2.imread(img)
        # get region
        if len(points) > 0:
            mask = np.zeros(im.shape, np.uint8)
            pts = np.array(points, np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], (1, 1, 1))
            im = im * mask
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.5
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            if cls == "person":
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                # for i in inds:
                #    print "class and num",CLASSES[i]
                #vis_detections(im, cls, dets, os.path.join('result', img.split('/')[-1]), thresh=CONF_THRESH)
               # if inds > 0:
                for box in dets[inds, :]:
                    cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 5)
                
                    cv2.putText(im,str('%.2f'%(box[4]*100)), (int(box[0]), int(box[1])),cv2.FONT_HERSHEY_COMPLEX,0.8, (0,0,255),thickness = 3, lineType = 4)
                return len(inds), dets[inds, :],im


if __name__ == '__main__':
    ped = Pedestrain()
    img_path = ['/home/wl/faster-rcnn-new/data/demo/test2.png','/home/wl/faster-rcnn-new/data/demo/test1.png']
    region = [{'X': 50, 'Y': 20}, {'X': 400, 'Y': 2}, {'X': 100, 'Y': 80}, {'X': 1, 'Y': 100}]
    region = []
    for img in img_path:
        im = cv2.imread(img)
        count,bboxes,im_pre = ped.get_pre(im, region)
        print "count =", count
        cv2.imshow('win', im_pre)
        cv2.waitKey()
