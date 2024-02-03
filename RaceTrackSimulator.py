import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pyprind

class RacetrackSimulator:
    def __init__(self):
        self.algorithms = {
            "1": ValueIterationAgent,
            "2": QLearningAgent,
            "3": SARSAAgent,
        }

    def run(self):
        def run(self):
            os.system("cls")  # clear the screen

        self.algorithms = {
            "1": self.value_iteration_train,
            "2": self.q_learning_train,
            "3": self.sarsa_train,
        }

        self.crash_policy = {"1": "nearest_position", "2": "starting_position"}

        self.tracks = {"1": "L-track.txt", "2": "O-track.txt", "3": "R-track.txt", "4": "W-track.txt"}

        self.iterations = {"1": 40, "2": 30000000, "3": 30000000, "4": 40}

        # ask user for the reactrack
        while True:
            track_choice = input(
                "\nPlease choose a track shape: "
                + "\n 1. L shaped track"
                + "\n 2. O shaped track"
                + "\n 3. R shaped track"
                + "\n 4. W shaped track\n>>> "
            )
            if track_choice in self.tracks.keys():
                break

        os.system("cls")  # clear the screen

        # ask user what algorithm to apply
        while True:
            self.algorithm = input(
                "\nSelect algorithm\n 1. Value iteration "
                + "\n 2. Q-learning \n 3. SARSA\n >>> "
            )
            if self.algorithm in self.algorithms.keys():
                break
        os.system("cls")

        # ask user for crashing policy
        while True:
            start_from_choice = input(
                "\nWhen the car crashes into a wall, return the car to: "
                + "\n 1. Nearest position on the track"
                + "\n 2. Starting position\n>>> "
            )
            if start_from_choice in self.crash_policy.keys():
                break
        os.system("cls")

        self.track_path = self.tracks[track_choice]
        self.load_track()  # load the racetrack
        self.episodes = self.iterations[self.algorithm]
        self.start_position()
        self.final_positions()

        self.Q = np.random.uniform(
            size=(
                *self.track.shape,
                len(self.velocities),
                len(self.velocities),
                len(self.actions),
            )
        )

        self.V = np.random.uniform(
            size=(*self.track.shape, len(self.velocities), len(self.velocities))
        )

        self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
        self.V[self.final[:, 0], self.final[:, 1], :, :] = 0

        print("Reinforcement Learning...")
        print("\nTrack:", self.track_path[:-4])
        self.crash_policy_name = (
            self.crash_policy[start_from_choice].replace("_", " ").title()
        )
        print("Crash policy:", self.crash_policy_name)
        self.start_from = self.crash_policy[start_from_choice]
        self.algorithms[self.algorithm]()
        self.make_policy()

        while True:
            do_simulation = input("\nSimulate the race (yes/no)? ")
            if do_simulation in ["yes", "no"]:
                if do_simulation == "yes":
                    self.print_racetrack_ = True
                    self.simulate()
                break

    def simulate(self):
        def simulate(self):
            steps_track = []
        max_steps = 250  # maximum number of steps
        for _ in range(50):
            self.start_position()
            self.v_y, self.v_x = (0, 0)
            steps = 0
            while True:
                steps += 1
                a = self.policy[(self.y, self.x, self.v_y, self.v_x)]
                self.update_state(a, self.action_probab)
                if self.print_racetrack_:
                    self.print_racetrack()
                # break the loop if the maximum number of steps is achieved
                if self.is_stuck() or steps > max_steps:
                    steps_track.append(max_steps)
                    break
                # break the loop if the car crossed the finish line
                if self.track[self.y, self.x] == "F":
                    steps_track.append(steps)
                    break
        self.number_of_steps.append(np.mean(steps_track))

    def print_racetrack(self):
        def print_racetrack(self):
            temp = self.track[self.y, self.x]  # current racetrack cell
        self.track[self.y, self.x] = "X"  # position of the car
        os.system("cls")  # clear the screen
        # print the racetrack
        for row in self.track:
            row_str = ""
            for char in row:
                row_str += f"{str(char):<1} ".replace(".", " ")
            print(row_str)
        self.track[self.y, self.x] = temp
        time.sleep(1)

    def learning_curve(self):
        def learning_curve(self):
            """method that creates a learning curve"""
        if self.algorithm == "1":
            x = range(len(self.number_of_steps))
        else:
            x = [50000 * i for i in range(len(self.number_of_steps))]
        y = self.number_of_steps

        # create a figure
        figure, ax = plt.subplots(figsize=(15, 5))
        ax.step(x, y, label=f"Crash policy:\n{self.crash_policy_name}")
        ax.plot(x, y, "ro", alpha=0.5)
        ax.set_ylim([0, 100])
        ax.grid(alpha=0.5)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Average number of steps finish the race", fontsize=12)
        ax.ticklabel_format(axis="x", style="sci")
        ax.xaxis.major.formatter._useMathText = True
        if self.algorithm == "1":
            name = "Value iteration algorithm"
        elif self.algorithm == "2":
            name = "Q-learning algorithm"
        else:
            name = "Sarsa algorithm"
        ax.set_title(name, fontsize=14)
        ax.legend(fontsize=12)
        # save figure
        plt.savefig(name + "-" + self.crash_policy_name + ".pdf")
        plt.show()

if __name__ == "__main__":
    simulator = RacetrackSimulator()
    simulator.run()