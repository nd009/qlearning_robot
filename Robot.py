import random

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5, verbose=False):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            self.alpha = 0
        else:
            self.t += 1
            self.epsilon = self.epsilon0*0.999**self.t
            self.alpha = 0.5

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """
        return self.maze.sense_robot()

    def create_Qtable(self, state):

        self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):

        def is_random_exploration():

            if random.random() > self.epsilon:
                return False
            else:
                return True

        if self.learning:
            if is_random_exploration():
                return random.choice(self.valid_actions)
            else:
                return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        elif self.testing:
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):

        if self.learning:
            self.Qtable[self.state][action] \
                = (1 - self.alpha) * self.Qtable[self.state][action] + \
                  self.alpha * (r + self.gamma * max(self.Qtable[next_state].values()))

    def update(self):

        self.state = self.sense_state()
        self.create_Qtable(self.state)

        action = self.choose_action()
        reward = self.maze.move_robot(action)

        next_state = self.sense_state()
        self.create_Qtable(next_state)

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state)
            self.update_parameter()

        return action, reward
