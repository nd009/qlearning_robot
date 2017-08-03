"""
Template for implementing QLearner  
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self,
        num_states=100,
        num_actions = 4,
        alpha = 0.2,
        gamma = 0.9,
        rar = 0.5,
        radr = 0.99,
        verbose = False):

        #TODO
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        #TODO
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print("s =", s,"a =",action)
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #TODO
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print("s =", s_prime,"a =",action,"r =",r)
        return action

