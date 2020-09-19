#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import random
import json

class ReplayBuffer:
    ## Only needs to implement insert with auto replacement
    ## No need to implement deletion
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.cur_size = 0
        self.next_index = 0

    def append(self, obj):
        self.buffer[self.next_index] = obj
        self.next_index = (self.next_index + 1) % self.max_size
        self.cur_size = min(self.max_size, self.cur_size + 1)

    def sample(self, batch_size):
        indices = random.sample(range(self.cur_size), batch_size)
        return [self.buffer[index] for index in indices]

    def save(self, path):
        pass