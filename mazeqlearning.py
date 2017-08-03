"""
Train a Q Learner in a navigation problem.
"""

import numpy as np
import random as rand
import QLearner as ql
import Maze


# convert the position to a single integer state
def to_state(pos):
    #TODO
    return 0


# train learner to go through maze multiple epochs
# each epoch involves one trip from start to the goal or timeout before reaching the goal
# return list of rewards of each trip
def train(maze, learner, epochs=500, timeout = 100000, verbose = False):
    #TODO
    rewards = np.zeros(epochs)
    return rewards


# run the code to train a learner on a maze
def maze_qlearning(filename):
    #TODO
    #initialize maze object
    #initialize learner object
    #execute train(maze, learner)
    #return median of all rewards

    return 0

if __name__=="__main__":
    rand.seed(5)
    maze_qlearning('testworlds/world01.csv')
