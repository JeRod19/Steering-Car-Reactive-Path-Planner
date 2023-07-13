# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:37:40 2022

@author: JRodr
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt

import math as m
import numpy as np

import Robots
import Enviroment as env
import Plans

class PID_Control():
    
    def __init__(self, p, i, d):
        self.p_gain = p
        self.i_gain = i
        self.d_gain = d
        
        self.cte = 0.0
        self.prev_cte = 0.0
        self.int_cte = 0.0
    
    def step(self, delta):
        self.cte = delta
        diff_cte = self.cte - self.prev_cte
        self.int_cte += self.cte
        
        self.prev_cte = self.cte
          
        return -self.p_gain * self.cte - self.d_gain * diff_cte - self.i_gain * self.int_cte
        
    def reset(self):
        self.cte = 0.0
        self.prev_cte = 0.0
        self.int_cte = 0.0
    
    def invert(self):
        self.cte *= -1
        self.prev_cte *= -1
        self.int_cte *= -1
    
    def set(self, p, i, d):
        self.p_gain = p
        self.i_gain = i
        self.d_gain = d

def check_future_collision(grid, robot, path, index):
    robot_pts =  np.linspace([-robot.width/2, 0], [robot.width/2, 0], int(robot.width))
    
    for i in range(10):
        init = path[index]
        end = path[index+1]
        
        path_orient = m.atan(init[0]/init[1])
        rotMat = np.array([[m.cos(path_orient), -m.sin(path_orient)],[m.sin(path_orient), m.cos(path_orient)]])
        
        check_pts = np.matmul(robot_pts, rotMat)
                
        for points in np.linspace(check_pts + init, check_pts + end, 10):
            for [Ry, Rx] in points:
                if Ry > 0 and Ry < len(grid) and \
                Rx > 0 and Rx < len(grid[0]) and \
                grid[int(Ry), int(Rx)] == 1:
                    return True
        index += 1    

        
        # for t in range(int(np.ceil(seg_length))):
        #     Rp = corners + u*t
        #     for corner in Rp:
        #         if corner[0] > 0 and corner[0] < len(grid) and \
        #         corner[1] > 0 and corner[1] < len(grid[0]) and \
        #         grid[int(corner[0]), int(corner[1])] == 1:
        #             return True
        
        
    return False

#Robot intial parameters
x = 200
y = 150
orientation = (m.pi * 45 / 180)

# Robot setup
width = 10
length = 20
steering = 0

#Movement parameters
time_limit = 10
dt = 0.1                    #step time = 0.1sec
velocity = 1.5                #Robot velocity = 5 cm/s
time_scale = 6.0
steps = int(time_limit / dt)

scale = [10, 10]

# Goal position and orientation
goal = [1200, 200, 0]

fig, ax = plt.subplots(1, 2, figsize =(18,8))

senseMap = env.enviroment('maps/ClearMap.jpg', scale)
realMap = env.enviroment('maps/RobotMap.jpg', scale)
   
robot = Robots.steering_car(length, width, realMap.pix_density)
robot.set(x/scale[1], y/scale[0], orientation)
# robot.set_noise( 0.1 *m.pi / 4, dt * velocity * 0.1, 2) 

realMap.update_screen(fig, ax[0], robot)
senseMap.update_screen(fig, ax[1], robot)

plan = Plans.plan_with_reverse(senseMap.grid, [goal[0]/scale[0], goal[1]/scale[1]], 2)

plan.compute_path(senseMap, robot, fig, ax[1])
path = plan.smooth(0.3, 0.3)


control = PID_Control(5.784263853039722, -0.04239115827521622, 14.551456094737718)

N = 0
index = 0

while not robot.check_goal(goal) and N < 800:
    
    y = robot.y
    x = robot.x
    robot_orient = robot.orientation
    
    #Control step
    # some basic vector calculations
    dx = path[index+1][1] - path[index][1]
    dy = path[index+1][0] - path[index][0]
    
    drx = x - path[index][1]
    dry = y - path[index][0]
    
    # u is the robot estimate projectes onto the path segment
    u = (drx * dx + dry * dy) / (dx * dx + dy * dy)
    
    # pick the next path segment     
    if u > 1.0 and index < len(path) - 1:
    # if x > path[index+1][1]  and index < len(path) - 1:
        index += 1
    if index == len(path) - 1:
        N = 300
        break
    
    d = ((path[index+1][0] - y)**2 + (path[index+1][1] - x)**2) ** 0.5
    index_increase = 0
    for i in range(2,3):
        if index + i < len(path) - 1:
            next_d = ((path[index+i][0] - y)**2 + (path[index+i][1] - x)**2) ** 0.5
            if next_d < d:
                index_increase = i
                next_d = d
    index += index_increase
    
    # the cte is the estimate projected onto the normal of the path segment
    dist_delta = (dry * dx - drx * dy) / (dx * dx + dy * dy)    
    robot_orient = robot.orientation
    path_orient = (m.atan2((path[index+2][0] - y),(path[index+2][1] - x)) + 2*m.pi) % (2*m.pi)
    
    orient_diff = ((robot_orient - path_orient) + m.pi) % (2*m.pi) - m.pi
    if abs(orient_diff) < 135 * m.pi / 180:
        steer = control.step(dist_delta)
        robot = robot.move(senseMap.grid, steer, velocity)
    else:
        steer = control.step(-dist_delta)
        robot = robot.move(senseMap.grid, steer, -velocity)

    
    something_new = robot.view(realMap.grid, senseMap.grid)
    if something_new:
        print("Something detected...")
        if check_future_collision(senseMap.grid, robot, path, index):
            path = plan.compute_path(senseMap, robot, fig, ax[1])
            path = plan.smooth(0.3, 0.3)
            index = 0
            print("Obstacle on path...")
        else:
            print("Path clear...")

    if N % 2 == 0:
        realMap.update_screen(fig, ax[0], robot, plan)
        ax[0].scatter(path[index+1,1],path[index+1,0], c = "red", s = 50)
        
        senseMap.update_screen(fig, ax[1], robot, plan)
        ax[1].scatter(path[index+1,1],path[index+1,0], c = "red", s = 50)
        plt.pause(0.000001)
    
    N += 1
    # print(robot, cte, index, u)

    



