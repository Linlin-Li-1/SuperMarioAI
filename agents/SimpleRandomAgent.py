#!/usr/bin/env python
# coding: utf-8

from agents.agent import Agent, MOVEMENTS
from random import choice

class SimpleRandomAgent(Agent):
    def __init__(self):
        super().__init__(MOVEMENTS.SIMPLE)
    def action(self, state):
        action = choice(self.movements)
        return action
    def feedback(self, states, reward, info, done):
        pass