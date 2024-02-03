import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pyprind

class SARSAAgent(RacetrackAgent):
    def __init__(self, v_min, v_max, gamma, action_probab, acceleration, learning_rate, episodes, iter_per_episode):
        super().__init__(v_min, v_max, gamma, action_probab, acceleration, learning_rate)
        self.episodes = episodes
        self.iter_per_episode = iter_per_episode

    def sarsa_train(self):
        """
        method that implements Sarsa learning algorithm
        """
        iter_per_episode = 20
        print("Algorithm NAME: SARSA")
        print("Episodes:", self.episodes)
        print("Iterations per episode:", iter_per_episode)
        print("Progressing:\n")

        bar = pyprind.ProgBar(self.episodes)
        for episode in range(self.episodes):

            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0

            # initialize state to arbitrary values
            y = np.random.choice(self.track.shape[0])
            x = np.random.choice(self.track.shape[1])
            v_y = np.random.choice(self.velocities)
            v_x = np.random.choice(self.velocities)

            a = np.argmax(self.Q[y, x, v_y, v_x])
            self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x

            for _ in range(iter_per_episode):
                if self.track[y, x] == "F" or self.track[y, x] == "#":
                    break

                # update state
                self.update_state(self.actions[a], self.action_probab)

                # choose the best action for a give state-action pair
                a_prime = np.argmax(self.Q[self.y, self.x, self.v_y, self.v_x])

                reward = -1
                self.Q[y, x, v_y, v_x, a] = (1 - self.learning_rate) * self.Q[
                    y, x, v_y, v_x, a
                ] + self.learning_rate * (
                                                    reward
                                                    + self.gamma * self.Q[self.y, self.x, self.v_y, self.v_x, a_prime]
                                            )
                y, x, v_y, v_x = self.y, self.x, self.v_y, self.v_x
                a = a_prime

            # make a simulation of the race
            if episode % 50000 == 0:
                self.make_policy()
                self.simulate()
            bar.update()

        print(bar)
