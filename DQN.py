#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np

class DQNBase:
    def __init__(self, input_shape, outlen, model = None):
        self.input_shape = input_shape
        self.outlen = outlen
        if model is None:
            self.build_model(input_shape, outlen)
        else:
            self.model = model
        self.compile_model()
    def build_model(self, input_shape, outlen):
        # To be implemented
        self.model = keras.Sequential()
        return self
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
        return self.__class__(self.input_shape, self.outlen, model = keras.models.clone_model(self.model))
    def predict(self, input, actions = None):
        mask = np.zeros((input.shape[0], self.outlen))
        if actions is None:
            return self.model.predict([input, mask], training = False)
        for id, num in enumerate(actions):
            mask[id, num] = 1
        return self.model.predict([input, mask], training = True)
    def fit(self, X, actions, Y, **args):
        mask = np.zeros((X.shape[0], self.outlen))
        for id, num in enumerate(actions):
            mask[id, num] = 1
        self.model.fit([X, mask], Y, **args)
        return self

# You can modify this class
class SampleDQN(DQNBase):
    def __init__(self, input_shape, outlen, model = None):
        super().__init__(input_shape, outlen, model)
    def build_model(self, input_shape, outlen):
        input_image = keras.Input(self.input_shape)
        input_mask = keras.Input(outlen)
        net_image = keras.Sequential([
            keras.layers.InputLayer(self.input_shape),
            keras.layers.Convolution2D(8, (5, 5), 3, activation = "relu"),
            keras.layers.SeparableConvolution2D(16, (3, 3), 1, activation = "relu"),
            keras.layers.SeparableConvolution2D(32, (3, 3), 1, activation = "relu"),
            keras.layers.SeparableConvolution2D(64, (3, 3), 1, activation = "relu"),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation = "relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(outlen, activation = "relu")
        ])
        output_image = net_image(input_image)
        output = keras.layers.multiply([output_image, input_mask])
        self.model = keras.Model([input_image, input_mask], output)
        return self
    def compile_model(self):
        self.model.compile(
            optimizer="adam",
            loss=keras.losses.Huber(),
            metrics=['mse'])
        return self