from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter

import cv2
import numpy as np
import math as m

import Robots
import Enviroment as env
import Plans

#Robot intial parameters
# x, y = [100, 300]
y,x = [100, 1550]

# Goal position and orientation
goal = [1200, 200]

orientation = (m.pi * 0 / 180)

width = 13
length = 26
steering = 4

#Movement parameters
time_limit = 10
dt = 0.1                    #step time = 0.1sec
velocity = 15                #Robot velocity = 5 cm/s
time_scale = 1.0
steps = int(time_limit / dt)

scale = [10, 10]


#####################################################################

if __name__ == "__main__":

    realMap = env.enviroment('RobotMap.jpg', scale)
    plt.imshow(realMap.grid, origin = 'lower')
    
    robot = Robots.steering_car(length, width, realMap.pix_density)
    robot.set(x/scale[1], y/scale[0], orientation)
    
    plan = Plans.plan(realMap.grid, [goal[0]/scale[0], goal[1]/scale[1]], 2)
    plan.compute_path(realMap, robot)
    
    plt.plot(plan.spath[:,1], plan.spath[:,0])
    # path = np.zeros((len(plan.spath), 2))
    # for i in range(len(plan.spath)):
    #     path[i,:] = plan.spath[i]
    
    # plt.plot(path[:,1], path[:,0])
