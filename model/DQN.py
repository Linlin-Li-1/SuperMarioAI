#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
from model.DQNBase import DQNBase

# You can modify this class
class SampleDQN(DQNBase):
    def __init__(self, input_shape, output_length, model = None):
        super().__init__(input_shape, output_length, model)
    def build_model(self, input_shape, output_length):
        input_image = keras.Input(self.input_shape)
        input_mask = keras.Input(output_length)
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
            keras.layers.Dense(output_length, activation = "relu")
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