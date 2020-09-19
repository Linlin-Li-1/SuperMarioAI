#!/usr/bin/env python
# coding: utf-8

import numpy as np

class figure:
    @classmethod
    def downsample(cls, s1 = (None, None, None), s2 = (None, None, None)):
        return lambda img: img[..., slice(*s1), slice(*s2), :]
    @classmethod
    def gray_scale(cls):
        return lambda img: np.mean(img, axis = -1).astype(np.uint8)
    @classmethod
    def channelize(cls):
        return lambda img: np.stack([i for i in img], axis = -1)
    @classmethod
    def normalize(cls):
        return lambda img: img / 255.0