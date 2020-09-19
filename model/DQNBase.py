#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np

class DQNBase:
    def __init__(self, input_shape, output_length, model = None):
        self.input_shape = input_shape
        self.output_length = output_length
        if model is None:
            self.build_model(input_shape, output_length)
        else:
            self.model = model
        self.compile_model()
    def build_model(self, input_shape, output_length):
        raise NotImplementedError()
    def compile_model(self):
        self.model.compile()
    def save_model(self, path):
        self.model.save_weights(path)
        return self
    def load_model(self, path):
        try:
            self.model.load_weights(path)
        except Exception as e:
            print(e)
        return self
    def clone_model(self):
        return self.__class__(self.input_shape, self.output_length, model = keras.models.clone_model(self.model))
    def predict(self, input, actions = None):
        mask = self._mask((input.shape[0], self.output_length), actions)
        return self.model.predict([input, mask])
    def fit(self, input, output, actions = None, **args):
        mask = self._mask((input.shape[0], self.output_length), actions)
        self.model.fit([input, mask], output, **args)
        return self
    def _mask(self, shape, actions = None):
        if actions is None:
            return np.ones(shape)
        mask = np.zeros(shape)
        for id, num in enumerate(actions):
            mask[id, num] = 1
        return mask