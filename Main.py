from matplotlib import pyplot as plt

import math as m
import numpy as np
from copy import deepcopy

import Robots
import Enviroment as env
import Plans 

def check_future_collision(grid, robot, path, index):
    robot = deepcopy(robot)
    currX = robot.x
    currY = robot.y

    remain_steps = len(path) - index -1
    steps_to_check = min(remain_steps, 20)

    for i in range(steps_to_check):
        [nextY, nextX, direct] = path[index+i]
        dx = nextX - currX
        dy = nextY - currY

        nextOrient = np.arctan2(dy , dx)
        if direct < 0:
            nextOrient -= np.pi
        if nextOrient < 0:
            nextOrient += 2 * np.pi
        
        robot.set(nextX, nextY, nextOrient)
        if robot.check_collision(grid) == False:
            return index+i
        
        currX = nextX
        currY = nextY
    
    return 0
   
#Robot intial parameters
[x, y, orientation] = [160, 160, (m.pi * 180 / 180)]
goal = [150, 2400, 0]

# Robot setup
width = 10
length = 20
steering = 0

#Movement parameters
time_limit = 10
dt = 0.1                    #step time = 0.1sec
velocity = 1.5                #Robot velocity = 5 cm/s
time_scale = 6.0
steps = 1

scale = [10, 10]

fig, ax = plt.subplots(1, 2, figsize =(18,8))

senseMap = env.enviroment('maps/ClearMap.jpg', scale)
realMap = env.enviroment('maps/RobotMap.jpg', scale)
   
robot = Robots.steering_car(length, width, realMap.pix_density)
robot.set(x/scale[1], y/scale[0], orientation)
robot.view(realMap.grid, senseMap.grid)

realMap.update_screen(fig, ax[0], robot)
senseMap.update_screen(fig, ax[1], robot)
plt.pause(0.001)

plan = Plans.plan_with_reverse(senseMap.grid, [goal[0]/scale[0], goal[1]/scale[1]], 2, robot.width)

path = plan.compute_path(senseMap, robot, fig, ax[1])

N = 0
index = 1
comingFrom = path[0]
while not robot.check_goal([goal[0]/scale[0], goal[1]/scale[1]], 5):
    
    movingTo = path[index]
    orientation = np.arctan2((movingTo[0] - comingFrom[0]) , (movingTo[1] - comingFrom[1]))
    if movingTo[2] < 0:
        orientation -= np.pi

    if orientation < 0:
        orientation += 2 * np.pi
    
    robot.set(movingTo[1], movingTo[0], orientation)
    something_new = robot.view(realMap.grid, senseMap.grid)

    comingFrom = movingTo

    realMap.update_screen(fig, ax[0], robot, plan)
    ax[0].scatter(path[index,1],path[index,0], c = "green", s = 50)
    
    senseMap.update_screen(fig, ax[1], robot, plan)
    ax[1].scatter(path[index,1],path[index,0], c = "green", s = 50)
    
    plt.pause(0.00001)
        
    if something_new:
        # print("Something detected...")
        obtsIdx = check_future_collision(senseMap.grid, robot, path, index)
        if (obtsIdx > 0):
            plan.update_heuristics(senseMap)
            path = plan.compute_path(senseMap, robot, fig, ax[1])
            index = 1

    
    index += 1

        
    

    



