import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pyprind

class RacetrackEnvironment:
    def __init__(self, track_path):
        self.track_path = track_path
        self.load_track()
        self.start_position()
        self.final_positions()

    def load_track(self):
        with open(self.track_path) as file:
            track_lines = file.readlines()[1:]
            self.track = np.asarray([list(line.strip("\n")) for line in track_lines])

    def start_position(self):
        start_positions = list(zip(*np.where(self.track == "S")))
        self.y, self.x = random.choice(start_positions)

    def final_positions(self):
        positions = list(zip(*np.where(self.track == "F")))
        self.final = np.asarray(positions)
