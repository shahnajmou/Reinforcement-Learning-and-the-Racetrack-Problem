import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pyprind

class ValueIterationAgent(RacetrackAgent):
    def __init__(self, v_min, v_max, gamma, action_probab, acceleration, learning_rate, episodes):
        super().__init__(v_min, v_max, gamma, action_probab, acceleration, learning_rate)
        self.episodes = episodes

    def value_iteration_train(self):
        """
        method that implements Value iteration algorithm
        """
        print("Algorithm NAME: Value Iteration")
        print("Iterations:", self.episodes)
        print("\nProgressing:\n")

        # initialize a progress bar object that allows visuzalization of the computation
        bar = pyprind.ProgBar(self.episodes)

        for iteration in range(self.episodes):
            # iterate over all possible states
            for y in range(self.track.shape[0]):
                for x in range(self.track.shape[1]):
                    for v_y in self.velocities:
                        for v_x in self.velocities:
                            if self.track[y, x] == "#":
                                self.V[y, x, v_y, v_x] = -10
                                continue

                            self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x

                            for a_index, a in enumerate(self.actions):
                                if self.track[y, x] == "F":
                                    self.reward = 0
                                else:
                                    self.reward = -1

                                self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x
                                # update state
                                self.update_state(a, 1)
                                new_state = self.V[self.y, self.x, self.v_y, self.v_x]

                                self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x
                                self.update_state((0, 0), 1)
                                new_state_failed = self.V[
                                    self.y, self.x, self.v_y, self.v_x
                                ]

                                expected_value = (
                                        self.action_probab * new_state
                                        + (1 - self.action_probab) * new_state_failed
                                )
                                self.Q[y, x, v_y, v_x, a_index] = (
                                        self.reward + self.gamma * expected_value
                                )

                            self.V[y, x, v_y, v_x] = np.max(self.Q[y, x, v_y, v_x])

            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            self.V[self.final[:, 0], self.final[:, 1], :, :] = 0
            self.make_policy()
            self.simulate()
            bar.update()

        print(bar)
