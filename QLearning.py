import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pyprind

class QLearningAgent(RacetrackAgent):
    def __init__(self, v_min, v_max, gamma, action_probab, acceleration, learning_rate, episodes, iter_per_episode):
        super().__init__(v_min, v_max, gamma, action_probab, acceleration, learning_rate)
        self.episodes = episodes
        self.iter_per_episode = iter_per_episode

    def q_learning_train(self):
        """
        method that implements Q-learning algorithm
        """
        # number of iterations per episode
        iter_per_episode = 20
        print("Algorithm NAME: Q-learning")
        print("Episodes:", self.episodes)
        print("Iterations per episode:", iter_per_episode)
        print("\nProgressing:\n")

        bar = pyprind.ProgBar(self.episodes)
        for episode in range(self.episodes):
            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            y = np.random.choice(self.track.shape[0])
            x = np.random.choice(self.track.shape[1])
            v_y = np.random.choice(self.velocities)
            v_x = np.random.choice(self.velocities)

            for _ in range(iter_per_episode):
                if self.track[y, x] == "F" or self.track[y, x] == "#":
                    break

                a = np.argmax(self.Q[y, x, v_y, v_x])
                self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x

                self.update_state(self.actions[a], self.action_probab)
                reward = -1

                # update the Q(s,a) values
                self.Q[y, x, v_y, v_x, a] = (1 - self.learning_rate) * self.Q[
                    y, x, v_y, v_x, a
                ] + self.learning_rate * (
                                                    reward
                                                    + self.gamma * np.max(self.Q[self.y, self.x, self.v_y, self.v_x])
                                            )

                y, x, v_y, v_x = self.y, self.x, self.v_y, self.v_x

            # make a simulation
            if episode % 50000 == 0:
                self.make_policy()
                self.simulate()
            bar.update()
        print(bar)
