import numpy as np
import random
import imageio

from multiprocessing import Pool, cpu_count

import tqdm
from os.path import join
from copy import deepcopy

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2

import matplotlib.pyplot as plt

class Runner(object):
    valid_actions = ['u', 'r', 'd', 'l'] # Up, Right, Down, Left
    robot_img = {d:imageio.imread(join("images/","robot-"+d+".jpg")) for d in valid_actions}
    logo_img = imageio.imread("images/logo.jpg")
    arrow_img = {d:imageio.imread(join("images/","arrow-"+d+".jpg")) for d in valid_actions}
    header_font = ImageFont.truetype("abel-regular.ttf", 60)
    font = ImageFont.truetype("abel-regular.ttf", 40)

    def __init__(self, robot, maze):

        self.maze = maze
        self.robot = robot

    def run_training(self, training_epoch, training_per_epoch=150, display_direction=False):

        self.train_robot_record = {}
        self.train_robot_statics = {
            'success': [],
            'reward': [],
            'times': [],
        }
        self.display_direction = display_direction
        # self.maze_data = {}

        def train_logger_before_act(e, i):

            self.train_robot_record[(e,i)] = {}
            self.train_robot_record[(e,i)]['id'] = (e,i)
            self.train_robot_record[(e,i)]['success'] = False

            self.train_robot_record[(e,i)]['state'] = self.robot.sense_state()
            self.train_robot_record[(e,i)]['qtable'] = self.robot.Qtable[self.robot.sense_state()].copy()

            self.train_robot_record[(e,i)]['epsilon'] = self.robot.epsilon
            self.train_robot_record[(e,i)]['alpha'] = self.robot.alpha
            self.train_robot_record[(e,i)]['gamma'] = self.robot.gamma

            self.train_robot_record[(e,i)]['maze_loc'] = self.maze.robot.copy()

            if self.display_direction:
                self.train_robot_record[(e,i)]['Qtable'] = deepcopy(self.robot.Qtable)

        def train_logger_after_act(e, i, action, reward):

            self.train_robot_record[(e,i)]['action'] = action
            self.train_robot_record[(e,i)]['reward'] = reward

        for e in range(training_epoch):
            accumulated_reward = 0
            run_time = 0
            for i in range(training_per_epoch):
                train_logger_before_act(e, i)
                action, reward = self.robot.update()
                train_logger_after_act(e, i, action, reward)
                run_time += 1
                accumulated_reward += reward
                if self.maze.robot['loc'] == self.maze.destination:
                    i+=1
                    train_logger_before_act(e,i)
                    self.train_robot_record[(e,i)]['success'] = True
                    break
            if self.maze.robot['loc'] == self.maze.destination:
                self.train_robot_statics['success'].append(1)
            else:
                self.train_robot_statics['success'].append(0)
            self.train_robot_statics['reward'].append(accumulated_reward)
            self.train_robot_statics['times'].append(run_time)
            self.maze.reset_robot()
            self.robot.reset()

    def run_testing(self, testing_per_epoch):

        self.test_robot_statics = {}
        self.test_robot_statics['success'] = []
        self.test_robot_statics['reward'] = []
        self.test_robot_statics['times'] = []

        self.robot.set_status(learning=False, testing=True)

        testing_per_epoch = int(self.maze.height * self.maze.height * 0.85)
        accumulated_reward = 0.
        run_time = 0
        for i in range(testing_per_epoch):
            run_time += 1
            _, reward = self.robot.update()
            accumulated_reward += reward
            if self.maze.robot['loc'] == self.maze.destination:
                break
        if self.maze.robot['loc'] == self.maze.destination:
            self.test_robot_statics['success'].append(1)
        else:
            self.test_robot_statics['success'].append(0)
        self.test_robot_statics['reward'].append(accumulated_reward)
        self.test_robot_statics['times'].append(run_time)

    # Generate video header
    def draw_header(self, base_image):
        logo_size = 200 # default sizes for logo is 200
        logo_image = np.vstack((Image.new('RGB', (200,50), color=(255,255,255)),self.logo_img))
        logo_image = np.vstack((logo_image,Image.new('RGB', (200,50), color=(255,255,255))))
        header_shape = (base_image.shape[1]-logo_size, logo_size+100) # width, height
        header_img = np.hstack((logo_image, Image.new('RGB', header_shape, color=(255,255,255))))
        return header_img

    # Draw robot on maze
    def draw_robot(self, base_image, parameters):
        img = base_image.copy()
        robot = parameters['maze_loc']
        grid_size = 100
        r,c = robot['loc']
        img[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :] += self.robot_img[robot['dir']]
        if self.display_direction:
            for state, q in parameters['Qtable'].items():
                r,c = state
                direction = max(q, key=q.get)
                img[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :] = \
                    (0.3*self.arrow_img[direction] + 0.7*img[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :]).astype('uint8')
        return img

    # Write on header
    def write_on_header(self, header_img, parameters):
        header = Image.fromarray(header_img.copy())
        draw = ImageDraw.Draw(header)
        line0 = "Epoch %d \nTrain %d \n"%(parameters['id'][0]+1, parameters['id'][1]+1)
        line1 = "Robot current state %s \n"%(str(parameters['state']))
        line2 = "q value:" + ','.join([a + ": %.2f"%q for a,q in parameters['qtable'].items()])
        if not parameters['success']:
            line3 = "\nAction %s, Reward received %f \n"%(parameters['action'],parameters['reward'])
            line4 = "Epsilon: %.2f, Alpha: %.2f, Gamma: %.2f"%(parameters['epsilon'],parameters['alpha'],parameters['gamma'])
        else:
            line3, line4 = '', ''
        draw.text((230, 85), line0, (0,0,0), font=self.header_font)
        draw.text((480, 50), line1+line2+line3+line4, (0,0,0), font=self.font)

        return np.array(header)

    def generate_movie(self, filename):

        base_image = self.maze.get_raw_maze_img()
        header_img = self.draw_header(base_image)

        def ensemble_image(train_id):
            new_header = self.write_on_header(header_img, self.train_robot_record[train_id])
            current_maze = self.draw_robot(base_image, self.train_robot_record[train_id])
            return (train_id, np.vstack((new_header, current_maze)))

        # For Multiprocessing, developing
        # with Pool(processes=max(cpu_count()-1,1)) as pool:
        #     with tqdm.tqdm(pool.imap(ensemble_image, runner.train_robot_record.keys()),
        #                    total=len(runner.train_robot_record.keys()), desc="Generating Images") as pbar:
        #         res = list(pbar)

        res = []
        with tqdm.tqdm(self.train_robot_record.keys(), desc="Generating Images") as pbar:
            for key in pbar:
                res.append(ensemble_image(key))

        height, width, _ = res[0][1].shape
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"XVID"), 10.0, (width//2,height//2))
        with tqdm.tqdm(sorted(res), desc="Generate Movies") as pbar:
            for (key,img) in pbar:
                writer.write(cv2.resize(img[:,:,::-1],(width//2,height//2)))

        writer.release()

    def plot_results(self):
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.title("Success Times")
        plt.plot(np.cumsum(self.train_robot_statics['success']))
        plt.subplot(132)
        plt.title("Accumulated Rewards")
        plt.plot(np.array(self.train_robot_statics['reward']))
        plt.subplot(133)
        plt.title("Runing Times per Epoch")
        plt.plot(np.array(self.train_robot_statics['times']))
        plt.show()
