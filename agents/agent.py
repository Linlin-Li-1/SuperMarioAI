#!/usr/bin/env python
# coding: utf-8

class MOVEMENTS:
    RIGHT_ONLY = range(5)
    SIMPLE = range(7)
    NO_UPDOWN = range(10)
    COMPLEX = range(12)
    
class Agent:
    def __init__(self, movements):
        self.movements = movements
    def reward(self, reward, info_old, info_new):
        return reward
    def action(self, states):
        raise NotImplementedError()
    def feedback(self, states, reward, info, done):
        raise NotImplementedError()