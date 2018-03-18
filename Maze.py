import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
from os.path import join

class Maze(object):
    """
    Maze objects have several main attributes:
    - maze_data: wall conditions in every cells are coded as a 4-bit number,
        with a bit value taking 0 if there is a wall and 1 if there is no wall.
        The 1s register corresponds with a square's top edge, 2s register the
        right edge, 4s register the bottom edge, and 8s register the left edge.
    """
    valid_actions = ['u', 'r', 'd', 'l'] # Up, Right, Down, Left
    direction_bit_map = {'u':1, 'r':2, 'd':4, 'l':8}
    # e.g., If there is an opening in the upside of the cell, the cell number & 1 is True
    move_map = {
        'u': (-1,0),
        'r': (0,+1),
        'd': (+1,0),
        'l': (0,-1),
    }
    action_unstability = {
        'u': {'l':0.1, 'u':0.8, 'r':0.1},
        'r': {'u':0.1, 'r':0.8, 'd':0.1},
        'd': {'r':0.1, 'd':0.8, 'l':0.1},
        'l': {'d':0.1, 'l':0.8, 'u':0.1},
    }
    robot_img = {d:imageio.imread(join("images/","robot-"+d+".jpg")) for d in valid_actions}

    def __init__(self, from_file=None, maze_size=None, trap_number=5, unstable_action=False):
        """
        You can construct a map from given file or just generating a random one.

        """
        if (from_file is not None) and (maze_size is None):
            with open(from_file, 'rb') as f_in:
                self.maze_data = np.genfromtxt(from_file,
                                          delimiter=',', dtype=np.uint16)
                # Check if the maze have inconsistency in some parts
                self.__validate_maze()
        elif maze_size is not None:
            self.__generate_maze(maze_size[0]*2+1, maze_size[1]*2+1)
        else:
            raise InputError("Invalid Input")

        self.height, self.width = self.maze_data.shape
        self.unstable_action = unstable_action

        # Generate trap and destination point of the maze
        self.__set_destination() # Only one destination
        self.__generate_trap(trap_number=trap_number) # Multiple traps

        self.__draw_raw_maze_img()

        self.__default_robot_loc = {
            'loc': (0,self.width-1),
            'dir': 'd',
        } # Default direction is down

        self.place_robot()
        self.set_reward()

    def __generate_maze(self, height=21, width=27, complexity=.25, density=.25):
        """
        Generate a random maze, based on:
        https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = random.randint(0, shape[1] // 2) * 2,  random.randint(0, shape[0] // 2) * 2
            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_,x_ = neighbours[random.randint(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        r,c = Z.shape

        # Convert to our maze style
        maze_data = np.zeros(((r-3)//2+1,(c-3)//2+1),dtype=np.uint8)
        for i in range(0,r-2,2):
            for j in range(0,c-2,2):
                maze_data[i//2,j//2] = sum([1,2,4,8][i] * ~block for i,block in enumerate(np.ravel(Z[i:i+3,j:j+3],order='F')[[3,7,5,1]]))

        self.maze_data = maze_data

    def __validate_maze(self):
        """
        Check if the input wall contains inconsistency
        """
        wall_errors = []
        height, width = self.maze_data.shape
        # Maze Size Check
        if height<=4 or width<=4:
            raise InputError("Input maze is too small")

        # Vertically Check
        for r in range(height-1):
            for c in range(width):
                if (self.maze_data[r,c] & 4 != 0) != (self.maze_data[r+1,c] & 1 != 0):
                    wall_errors.append([(r,c), 'v'])
        # Horizontally Check
        for r in range(height):
            for c in range(width-1):
                if (self.maze_data[r,c] & 2 != 0) != (self.maze_data[r,c+1] & 8 != 0):
                    wall_errors.append([(r,c), 'h'])
        # Output Errors
        if wall_errors:
            for cell, wall_type in wall_errors:
                if wall_type == 'v':
                    cell2 = (cell[0]+1, cell[1])
                    print('Inconsistent vertical wall betweeen {} and {}'.format(cell, cell2))
                else:
                    cell2 = (cell[0], cell[1]+1)
                    print('Inconsistent horizontal wall betweeen {} and {}'.format(cell, cell2))
            raise Exception('Consistency errors found in wall specifications!')

    def __set_destination(self, destination_coord=None):
        """
        Set destination coordinates, default in center
        """
        if not destination_coord:
            destination_coord = (self.height//2,self.width//2)
        self.destination = destination_coord

    def __generate_trap(self, trap_number=5):
        """
        Randomly generate traps
        """
        if trap_number > self.width * self.height*0.1:
            raise ValueError('Too many traps for such small maze')

        # Avoid repeated traps
        destination = int(self.destination[0] * self.width + self.destination[1])
        valid_range = list(range(1,destination)) + list(range(destination+1,int((self.width-1)*(self.height-1))))
        trap_list = random.sample(valid_range,trap_number)
        self.__traps = [(ele//self.width, ele%self.width) for ele in trap_list]

    def __draw_raw_maze_img(self):
        # Load grid images
        grid_images = []
        for i in range(16):
            grid_images.append(imageio.imread(join("images/",str(i)+".jpg")))
        maze = np.vstack((np.hstack((grid_images[i] for i in row)) for row in self.maze_data))

        # Display traps and destination
        trap_img = imageio.imread(join("images","trap.jpg"))
        dest_img = imageio.imread(join("images","destination.jpg"))
        grid_size = 100 # default sizes for grid, trap and destination are 100
        for (r,c) in self.__traps:
            maze[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :] += trap_img
        r,c = self.destination
        maze[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :] += dest_img

        # Final maze image
        self.__raw_maze_img = maze

    def get_raw_maze_img(self):

        return self.__raw_maze_img.copy()

    def draw_current_maze(self):
        grid_size = 100 # default sizes for grid, trap and destination are 100
        logo_size = 200 # default sizes for logo is 200
        r,c = self.robot['loc']
        current_maze_img = self.__raw_maze_img.copy()
        current_maze_img[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :] += \
            self.robot_img[self.robot['dir']]
        return current_maze_img

    def __repr__(self):
        plt.figure(figsize=(self.height,self.width))
        plt.imshow(self.draw_current_maze())
        plt.axis('off')
        plt.show()
        return 'Maze of size (%d, %d)'%(self.height, self.width)

    def is_permissible(self, location, direction):
        """
        Returns a boolean designating whether or not a cell is passable in the
        given direction. Cell is input as a tuple. Directions is input as single
        letter 'u', 'r', 'd', 'l'.
        """
        try:
            return (self.maze_data[location] & self.direction_bit_map[direction])!=0
        except:
            print('Invalid direction or location provided!')

    def place_robot(self, robot_loc=None):
        """
        Place robot into the maze, default in (0,0)
        """
        if not robot_loc:
            robot_loc = self.__default_robot_loc.copy()
        self.robot = robot_loc

    def set_reward(self):
        """
        Set rewards for different situations.
        """
        self.reward = {
            "hit_wall": -10.,
            "destination": 50.,
            "trap": -30.,
            "default": -0.1,
        }

    def move_robot(self, direction):
        """
        Move the robot location according to its location and direction
        Return the new location and moving reward
        """
        # Random choose action due to action unstability

        if not direction in self.valid_actions:
            raise ValueError("Invalid Actions")

        if self.unstable_action:
            unstable_act = self.action_unstability[direction]
            direction = np.random.choice(unstable_act.keys(), p=unstable_act.values())

        if self.is_permissible(self.robot['loc'],direction):
            self.robot['loc'] = tuple((i+di for i,di in zip(self.robot['loc'],self.move_map[direction])))
            self.robot['dir'] = direction
            if self.robot['loc'] in self.destination:
                reward = self.reward['destination']
            elif self.robot['loc'] in self.__traps:
                reward = self.reward['trap']
            else:
                reward = self.reward['default']
        else:
            self.robot['dir'] = direction
            reward = self.reward['hit_wall']
        return reward

    def sense_robot(self):

        return self.robot['loc']

    def reset_robot(self):

        self.robot = self.__default_robot_loc.copy()
