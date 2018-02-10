#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: coco.py

import os
import config
from nuclei import NucleiDataset

from tensorpack.utils.timer import timed_operation


__all__ = ['COCODetection', 'COCOMeta']

COCO_NUM_CATEGORY = 1
config.NUM_CLASS = COCO_NUM_CATEGORY + 1


class _COCOMeta(object):
    INSTANCE_TO_BASEDIR = {
        'stage1_train': 'stage1_train',
        'stage1_test': 'stage1_test'
    }

    def valid(self):
        return hasattr(self, 'cat_names')

    def create(self, cat_ids, cat_names):
        """
        cat_ids: list of ids
        cat_names: list of names
        """
        assert not self.valid()
        assert len(cat_ids) == COCO_NUM_CATEGORY and len(cat_names) == COCO_NUM_CATEGORY
        self.cat_names = cat_names
        self.class_names = ['BG'] + self.cat_names

        # background has class id of 0
        self.category_id_to_class_id = {
            v: i + 1 for i, v in enumerate(cat_ids)}
        self.class_id_to_category_id = {
            v: k for k, v in self.category_id_to_class_id.items()}
        config.CLASS_NAMES = self.class_names


COCOMeta = _COCOMeta()


class COCODetection(object):
    def __init__(self, basedir, name, mode=None):
        """
        mode: train or val
        """
        assert name in COCOMeta.INSTANCE_TO_BASEDIR.keys(), name
        self.name = name
        # data/stage1_train
        self._imgdir = os.path.join(basedir, COCOMeta.INSTANCE_TO_BASEDIR[name])
        assert os.path.isdir(self._imgdir), self._imgdir

        # initialize the meta
        cat_ids = [1]
        cat_names = ['nuclei']
        if not COCOMeta.valid():
            COCOMeta.create(cat_ids, cat_names)
        else:
            assert COCOMeta.cat_names == cat_names
            
        self.coco = NucleiDataset(self._imgdir, mode)

    def load(self):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            
            # list of dict, each has keys: 
            #id
            #file_name: data/stage1_train/dec1764c00e8b3c4bf1fc7a2fda341279218ff894186b0c2664128348683c757/images/dec1764c00e8b3c4bf1fc7a2fda341279218ff894186b0c2664128348683c757.png
            #mask_dir: data/stage1_train/dec1764c00e8b3c4bf1fc7a2fda341279218ff894186b0c2664128348683c757/masks/
            #name: dec1764c00e8b3c4bf1fc7a2fda341279218ff894186b0c2664128348683c757
            
            # determined after load image and masks:
            #height,width
            #boxes nx4
            #class n, always >0
            #is_crowd # n
            imgs = self.coco.image_info
    
            for img in imgs:
                img['file_name'] = img['path'] # abosolute file name.
            return imgs

    @staticmethod
    def load_many(basedir, names, mode=None):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            coco = COCODetection(basedir, n, mode=None)
            ret.extend(coco.load())
        return ret


if __name__ == '__main__':
    c = COCODetection('data', 'stage1_train', 'train')
    imgs = c.load()
    print("#Images:", len(imgs))
          
    c = COCODetection('data', 'stage1_train', 'val')
    imgs = c.load()
    print("#Images:", len(imgs))
