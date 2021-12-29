import numpy as np
import tensorflow as tf
import cv2
import object_detection.visualization_utils as vis_util

def create_category_index(label_path='coco_ssd_mobilenet/labelmap.txt'):
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index

