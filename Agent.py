import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pyprind

class RacetrackAgent:
    def __init__(self, v_min, v_max, gamma, action_probab, acceleration, learning_rate):
        self.actions = [(i, j) for j in acceleration for i in acceleration]
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.velocities = np.arange(v_min, v_max + 1, 1)
        self.action_probab = action_probab
        self.learning_rate = learning_rate
        self.threshold = 0.02
        self.number_of_iterations = 50
        self.y_stuck = 0
        self.x_stuck = 0
        self.stuck_counter = 0
        self.print_racetrack_ = False
        self.number_of_steps = []

    def update_velocities(self, action):
        v_y_temp = self.v_y + action[0]
        v_x_temp = self.v_x + action[1]

        if abs(v_x_temp) <= self.v_max:
            self.v_x = v_x_temp
        if abs(v_y_temp) <= self.v_max:
            self.v_y = v_y_temp

    def within_track(self):
        return 0 <= self.y < self.track.shape[0] and 0 <= self.x < self.track.shape[1]

    def update_state(self, action, probability):
        if np.random.uniform() < probability:
            self.update_velocities(action)

        y_temp, x_temp = self.y, self.x
        self.x += self.v_x
        self.y += self.v_y

        if self.within_track() and self.track[self.y, self.x] != "#":
            if self.v_y == 0:
                if "#" in self.track[y_temp, min(self.x, x_temp): max(self.x, x_temp)].ravel():
                    self.x = x_temp
                    self.v_y, self.v_x = 0, 0
            elif self.v_x == 0:
                if "#" in self.track[min(self.y, y_temp): max(self.y, y_temp), self.x].ravel():
                    self.y = y_temp
                    self.v_y, self.v_x = 0, 0
            elif self.v_x == self.v_y:
                if "#" in self.track[min(self.y, y_temp): max(self.y, y_temp),
                          min(self.x, x_temp): max(self.x, x_temp)]:
                    self.x, self.y = x_temp, y_temp
                    self.v_y, self.v_x = 0, 0
            else:
                if "#" in self.track[min(self.y, y_temp): max(self.y, y_temp),
                          min(self.x, x_temp): max(self.x, x_temp)].ravel():
                    self.x, self.y = x_temp, y_temp
                    self.v_y, self.v_x = 0, 0

        if not self.within_track() or self.track[self.y, self.x] == "#":
            self.return_to_track()

    def return_to_track(self):
        if self.start_from == "nearest_position":
            self.x += -self.v_x
            self.y += -self.v_y
            L = [1] * abs(self.v_x) + [0] * (2 * abs(self.v_y))
            for i in L:
                if i:
                    self.x += np.sign(self.v_x)
                    if self.within_track() and self.track[self.y, self.x] == "#":
                        self.x += -np.sign(self.v_x)
                        break
                else:
                    self.y += np.sign(self.v_y)
                    if self.within_track() and self.track[self.y, self.x] == "#":
                        self.y += -np.sign(self.v_y)
                        break
        elif self.start_from == "starting_position":
            self.start_position()
        self.v_y, self.v_x = 0, 0

    def is_stuck(self):
        if self.y_stuck == self.y and self.x_stuck == self.x:
            self.stuck_counter += 1
            self.y_stuck = self.y
            self.x_stuck = self.x
            return self.stuck_counter >= 4
        else:
            self.stuck_counter = 0
            self.y_stuck = self.y
            self.x_stuck = self.x
        return False