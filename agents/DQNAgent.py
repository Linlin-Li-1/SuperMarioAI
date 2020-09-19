#!/usr/bin/env python
# coding: utf-8

from agents.agent import Agent, MOVEMENTS
from model.DQN import SampleDQN
from replaybuffer import ReplayBuffer
from figure import figure
import numpy as np
import random
import json

class DQNAgent(Agent):
    class SAVE:
        MEMORY = 1
        TARGETNETWORK = 2
        TRAINNETWORK = 4
        HYPERPARAM = 8
        ALL = 15
    def __init__(self, DQNType, input_shape, replaybuffersize = 100000, input_preprocess = []):
        super().__init__(MOVEMENTS.COMPLEX)
        self.memory = ReplayBuffer(replaybuffersize)
        self.train_network = DQNType(input_shape, len(self.movements))
        self.target_network = self.train_network.clone_model()
        self.input_preprocess = input_preprocess

        ## Initialize
        self.counter = 0
        self.epsilon = 1

        ## hyperparameters
        self.hyperparams = {
            "burn_in" : 10000,
            "copy_each" : 5000,
            "learn_each" : 1,
            "save_each" : 5000,
            "final_epsilon": 0.1,
            "epsilon_decay_rate": 0.99998,
            "batch_size" : 32,
            "gamma" : 0.99
        }
    def setparam(self, **kwargs):
        for key, val in kwargs.items():
            self.hyperparams[key] = val
        return self
    def getparams(self):
        return self.hyperparams
    def preprocess(self, image):
        for pc in self.input_preprocess:
            image = pc(image)
        return image
    def reward(self, reward, info_old, info_new):
        return reward + (info_new["score"] - info_old["score"]) / 100
    def action(self, states):
        self.action_states = self.preprocess(states)
        ## Random exploration
        if random.uniform(0,1) < self.epsilon:
            self.action_num = random.choice(range(len(self.movements)))
        ## Make a decision based on the network
        else:
            normalized_states = figure.normalize()(self.action_states) # convert to 0-1 scale
            output = self.train_network.predict(normalized_states[None, ...])
            self.action_num = np.argmax(output)
        return self.movements[self.action_num]
    def feedback(self, states, reward, info, done):
        # what to do after getting a reward
        self.counter += 1
        self.memory.append((
            self.action_states,  # already preprocessed
            self.action_num, 
            reward,
            info,
            done,
            self.preprocess(states)
        ))
        self.updateNetwork()
    def save(self, file_path, saveMethod = None):
        if saveMethod is None:
            saveMethod = self.SAVE.ALL
        if (saveMethod & self.SAVE.MEMORY):
            self.memory.save(file_path + "memory")
        if (saveMethod & self.SAVE.TARGETNETWORK):
            self.target_network.save_model(file_path + "target_net")
        if (saveMethod & self.SAVE.TRAINNETWORK):
            self.train_network.save_model(file_path + "train_net")
        # if (saveMethod & self.SAVE.HYPERPARAM):
        #     with open(file_path + "hyperparam.json", "w") as f:
        #         json.dump(self.hyperparams, f, indent=2) 
    def load(self, file_path):
        try:
            self.target_network.load_model(file_path + "target_net")
            self.train_network.load_model(file_path + "train_net")
            with open(file_path + "hyperparam.json", "r") as f:
                self.hyperparams = json.load(f) 
        except Exception as e:
            print(e)
    def updateNetwork(self):
        if self.counter < self.hyperparams["burn_in"]:
            return
        self.epsilon *= self.hyperparams["epsilon_decay_rate"]
        self.epsilon = max(self.epsilon, self.hyperparams["final_epsilon"])
        if (self.counter - self.hyperparams["burn_in"]) % self.hyperparams["learn_each"] == 0:
            self.learn()
        if (self.counter - self.hyperparams["burn_in"]) % self.hyperparams["copy_each"] == 0:
            self.target_network = self.train_network.clone_model()
        if (self.counter - self.hyperparams["burn_in"]) % self.hyperparams["save_each"] == 0:
            self.save("./autosave/step_" + str(self.counter))
    def learn(self):
        learn_sample = self.memory.sample(self.hyperparams["batch_size"])
        state_raw = np.stack([states for states, _, _, _, _, _ in learn_sample], axis = 0)
        actions = [action for _, action, _, _, _, _ in learn_sample]
        rewards = [reward for _, _, reward, _, _, _ in learn_sample]
        not_done = [not done for _, _, _, _, done, _ in learn_sample]
        next_state_raw = np.stack([states for _, _, _, _, _, states in learn_sample], axis = 0)
        state = figure.normalize()(state_raw)
        next_state = figure.normalize()(next_state_raw)
        best_action_next = np.argmax(self.train_network.predict(next_state), axis = 1)
        # Predicts the Q values calculated at the best_action_next
        # We shall only keep those entries corresponding to the real actions taken
        # Terminal states should not involve calculating the expected Q value.
        Q_value_next_target_mat = self.target_network.predict(next_state, actions = best_action_next)
        Q_value_next_target_vec = np.max(Q_value_next_target_mat, axis = 1)
        Q_value_target_vec = np.array(rewards) + self.hyperparams["gamma"] * np.array(not_done) * Q_value_next_target_vec
        Q_value_target_mat = np.zeros(Q_value_next_target_mat.shape)
        for id, num in enumerate(actions):
            Q_value_target_mat[id, num] = Q_value_target_vec[id]
        
        self.train_network.fit(state, actions, Q_value_target_mat, verbose=0)