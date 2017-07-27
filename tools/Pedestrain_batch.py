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
#           'aeroplane', 'bicycle', 'bird', 'boat',
 #          'bottle', 'bus', 'car', 'cat', 'chair',
 #          'cow', 'diningtable', 'dog', 'horse',
 #          'motorbike', 'person', 'pottedplant',
 #          'sheep', 'sofa', 'train', 'tvmonitor')


def vis_detections(im, class_name, dets, img_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(img_name)

class Pedestrain(object):
    def __init__(self, use_cpu=False, gpu_dev=0):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        self.prototxt = os.path.join(cfg.MODELS_DIR, 'VGG16',
                                     'faster_rcnn_end2end', 'test.prototxt')
        #self.caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
         #                              'VGG16_faster_rcnn_final.caffemodel')
	self.caffemodel = '/home/wl/faster-rcnn-new/output/faster_rcnn_end2end/voc_2007_train_inout/vgg16_faster_rcnn_iter_1000.caffemodel'
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

    def get_count(self, img_path, region):
        """Detect object classes in an image using pre-computed object proposals."""
        # transform region
        points = []
        for xy in region:
            points.append([xy['X'], xy['Y']])

        # Load the demo image
        im_file = os.path.join(cfg.DATA_DIR, 'demo', img_path)
        im = cv2.imread(im_file)

        # get region
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
                return len(inds)



    def get_pre(self, img):
        # Load the demo image

        im = cv2.imread(img)

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
                vis_detections(im, cls, dets, os.path.join('result',img.split('/')[-1]), thresh=CONF_THRESH)
                return len(inds), dets[inds, :]


if __name__ == '__main__':
    ped = Pedestrain()
    root_dir = '/home/wl/faster-rcnn-new'
    f = open(os.path.join(root_dir, 'test.txt'), 'r')
    lines = f.readlines()
    f.close()
    indexs = [os.path.join(root_dir, 'inout_test', x.strip() + '.jpg') for x in lines]
    # region = [{'X': 50, 'Y': 20}, {'X': 400, 'Y': 2}, {'X': 300, 'Y': 200}, {'X': 1, 'Y': 400}]
    results = open('results', 'w')
    for img in indexs:
        print 'image:', img
        count, bboxes = ped.get_pre(img)
        results.write(img + '\n')
        results.write(str(count) + '\n')
        for box in bboxes:
            print 'box:', box
            results.write(
                str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(box[4]) + '\n')

