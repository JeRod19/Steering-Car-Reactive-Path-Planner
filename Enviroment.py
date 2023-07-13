import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

import cv2 
import math as m

class enviroment:
    
    def __init__(self, path, scale):
        env = cv2.imread(path, 0)
        env = 1 - np.round(env / 255.0)
        
        self.map = env
        
        self.scale = scale
        self.grid_shape = [env.shape[0] / self.scale[0], env.shape[1] / self.scale[1]]
        self.grid = np.zeros((int(np.ceil(self.grid_shape[0])), int(np.ceil(self.grid_shape[1]))))
        
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                row2 = self.scale[0] * row
                col2 = self.scale[1] * col
                val = round(self.map[row2:row2+self.scale[0], col2:col2+self.scale[1]].sum() / (self.scale[0] * self.scale[1]))
                self.grid[row,col] = val
    
        
        self.height = 400.0
        self.width = 450.0
        self.pix_density = self.grid_shape[1] / self.width 
        
        # self.figure, self.ax = plt.subplots(figsize =(7,7))
        self.rect = None
        self.tria = None        
        
    def patch_robot(self, robot, ax):
        x = robot.x
        y = robot.y
        th = 180 * robot.orientation / m.pi
        widthR = robot.width
        lengthR = robot.length
        
        #Robot body
        corner = [-widthR / 2, -lengthR / 2]
        rotMat = np.array([[m.cos(robot.orientation), -m.sin(robot.orientation)],[m.sin(robot.orientation), m.cos(robot.orientation)]])
        
        corner = np.matmul(corner, rotMat) + [y, x]
        self.rect = patches.Rectangle((corner[1], corner[0]), lengthR, widthR, th, edgecolor='r', facecolor="r", alpha = 0.8)
        
        #View range
        rotMat = np.array([[m.cos(robot.orientation), -m.sin(robot.orientation)],[m.sin(robot.orientation), m.cos(robot.orientation)]])
        
        camara = np.matmul(robot.camara, rotMat)
        camara += [y, x]
        
        horizon = np.matmul(robot.horizon, rotMat)
        horizon += [y, x]

        
        xy = [[camara[1], camara[0]], [horizon[0,1], horizon[0,0]], [horizon[-1,1], horizon[-1,0]]]
        self.tria = patches.Polygon(xy, closed=True, edgecolor='g', facecolor="g", alpha = 0.5)
        
        ax.add_patch(self.rect)
        ax.add_patch(self.tria)
    
    def plot_plan(self, plan, ax):
        path = plan.path
        plt.plot(path[:,1], path[:,0], '-*')
    
    def update_screen(self, fig, ax, robot = None, plan = None):
        plt.figure(fig.number)
        plt.sca(ax)
        
        ax.cla()
        
        ax.set_ylim(0, len(self.grid))
        ax.set_xlim(0, len(self.grid[0]))
        
        ax.imshow(self.grid, origin = 'lower')
        
        if not robot == None:
            self.patch_robot(robot, ax)
        
        if not plan == None:
            self.plot_plan(plan, ax)
        
        
 