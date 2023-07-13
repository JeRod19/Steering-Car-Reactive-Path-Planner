from matplotlib import pyplot as plt

import math as m

import Robots
import Enviroment as env


       

#Robot intial parameters
x = 200
y = 1200
orientation = (m.pi * 0 / 180)

width = 10
length = 20
steering = 4

#Movement parameters
time_limit = 10
dt = 0.1                    #step time = 0.1sec
velocity = 15                #Robot velocity = 5 cm/s
time_scale = 1.0
steps = int(time_limit / dt)

scale = [10, 10]

# Goal position and orientation
Goal = [300, 800, 0]

senseMap = env.enviroment('ClearMap.jpg', scale)
realMap = env.enviroment('RobotMap.jpg', scale)
   
robot = Robots.steering_car(length, width, realMap.pix_density)
robot.set(x/scale[1], y/scale[0], orientation)

fig, ax = plt.subplots(1, 2, figsize =(18,8))

for i in range(steps):
    
    robot = robot.move(realMap.grid, m.pi * steering / 180, dt * velocity)
    robot.view(realMap.grid, senseMap.grid)
    # steering = random.gauss(steering, 8)
    steering *= 1.0001
    
    realMap.update_screen(robot, fig, ax[0])
    plt.pause(dt /(2 * time_scale))
    senseMap.update_screen(robot, fig, ax[1])
    plt.pause(dt /(2 * time_scale))
    
    if robot.num_collisions > 0:
        break
    


