#!/usr/bin/env python
# coding: utf-8

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

class Environment:
    MOVEMENTS = COMPLEX_MOVEMENT
    def __init__(self, world, stage, version):
        self.game = 'SuperMarioBros-'+ str(world) + '-'+ str(stage) + '-v' + str(version)
        self.settings = {
            "SkipFrame":0, # Skip several frames between interaction with the agent
            "RenderScreen":0, # Render screen every several frames, 0 means no render at all.
            "PrintInformation":True, # Print information after each episode.
            "ShowEpisodeEach":1
        }
    def setparam(self, **kwargs):
        for name, value in kwargs.items():
            self.settings[name] = value
        return self
    def getparams(self):
        return self.settings
    def run(self, agent, maxepisode = -1):
        ## initialize environment
        env = JoypadSpace(gym_super_mario_bros.make(self.game), Environment.MOVEMENTS)
        ## initialize parameters (all variables without `acc` will be set to zero at the beginning of an episode)
        episode = -1
        frame = 0 # Current frame 
        step = acc_step = 0 # The agent makes decisions on each step instead of each frame.
                            # A step contains multiple frames (= `SkipFrame` + 1).
        reward = raw_reward = acc_reward = 0
        total_acc_reward = 0
        max_acc_reward = float("-inf") # The total and maximum rewards obtained in an episode.
        done = True # Whether this episode is done (lose or win)
        info = old_info = None # More information of current status

        FramePerStep = self.settings["SkipFrame"] + 1

        time_world_start = time.time()
        time_episode_start = time_print_start = None
        

        while True:
            if done:
                time_episode_end = time.time()
                if episode >= 0 and self.settings["PrintInformation"]:
                    print(
                        f"Episode {episode} ended in {step} steps. "
                        f"Average fps: {frame/(time_episode_end - time_episode_start)}. " 
                        f"Reward: {acc_reward}. "
                        f"Position: {info.get('x_pos', None)}. "
                    )
                time_episode_start = time_episode_end
                states = env.reset()[None,...].repeat(FramePerStep, axis = 0)
                # env.reset() resets the environment and returns the image of the first frame on screen
                # An image is a 3d array, and here states is a 4d array where `FramePerStep` images are stacked.
                # states = [Image_0, Image_1, ..., Image_{FramePerStep - 1}]
                total_acc_reward += acc_reward
                max_acc_reward = max(max_acc_reward, acc_reward)
                step = reward = frame = acc_reward = 0
                episode += 1
                time_episode_start = time.time()

            if episode == maxepisode:
                break

            ## make decision every other SkipFrame steps
            if frame % FramePerStep == 0: 
                action = agent.action(states)
                step += 1
                acc_step += 1

            ## repeat the same action between two decisions
            _state, _reward, _done, _info = env.step(action)
            states[frame % FramePerStep,...] = _state
            raw_reward += _reward / FramePerStep
            done = _done
            info = _info
            if old_info is None:
                old_info = _info
            
            ## call back for the Agent to update weights
            if (frame and frame % FramePerStep == 0) or done:
                # customize reward through agent's `reward` function 
                reward = agent.reward(raw_reward, old_info, info)
                # tell the agent of current state, reward and other information 
                agent.feedback(states, reward, info, done)
                acc_reward += reward
                raw_reward = 0
                old_info = info

            frame += 1

            ## Render screen (subject to change)
            if self.settings["RenderScreen"] > 0 and episode % self.settings["ShowEpisodeEach"] == 0 and frame % self.settings["RenderScreen"] == 0:
                if time_print_start is None:
                    time_print_start = 0
                else:
                    time_print_start = time_print_end
                time_print_end = time.time()
                fps = self.settings["RenderScreen"] / (time_print_end - time_print_start)
                try:
                    plt.figure(figsize = (6, 6))
                    plt.clf()
                    fig_raw = env.render(mode='rgb_array')
                    plt.imshow(fig_raw)
                    plt.title("Episode: %d. Frame: %d. Fps: %.3f.\n"
                                "Reward: %.2f. Mean: %.2f. Max: %.2f\n"
                                "Position: (%d, %d). Total Steps: %d.\n"
                                "Time: %.2fs (episode), %.2fs (total)."
                        % (episode, frame, fps, acc_reward, 
                        0 if episode == 1 else total_acc_reward / (episode - 1),
                        max_acc_reward,
                        _info["x_pos"], _info["y_pos"], acc_step, 
                        time_print_end - time_episode_start, time_print_end - time_world_start))
                    plt.axis('off')
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                except KeyboardInterrupt:
                    break
        env.close()
        print(f"Total time used: {time.time() - time_world_start}s")



