#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py

import numpy as np
import cv2

from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import transform
from tensorpack.utils import logger

import pycocotools.mask as cocomask

import config


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def size(self):
        return self._size

    def get_data(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class CustomResize(transform.TransformAugmentorBase):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, size, max_size, interp=cv2.INTER_LINEAR):
        """
        Args:
            size (int): the size to resize the shortest edge to.
            max_size (int): maximum allowed longest edge.
        """
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return transform.ResizeTransform(h, w, newh, neww, self.interp)


def box_to_point8(boxes):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def segmentation_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def print_config():
    logger.info("Config: ------------------------------------------")
    for k in dir(config):
        if k == k.upper():
            logger.info("{} = {}".format(k, getattr(config, k)))
    logger.info("--------------------------------------------------")


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (x1y1x2y2)]. matterport use (y1, x1, y2, x2)
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.float32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
#            x1 -= 1 #! is this useful?
#            y1 -= 1
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.float32)