from matplotlib import pyplot as plt

import cv2
import numpy as np
import math as m
from copy import deepcopy


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
        
def run(argrobot, p, show = False, n=100):
    
    robot = deepcopy(argrobot)
    control = PID_Control(p[0], p[1], p[2])
    
    N = 0
    index = 0
    err = 0
    trajectory = []
    while not robot.check_goal([goal[0]/scale[0], goal[1]/scale[1]]) and N < n:
        y = robot.y
        x = robot.x
        robot_orient = robot.orientation
        
        trajectory.append([y, x, path[index+1][0], path[index+1][1]])
        
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
            N = n
            break
        
        d = ((path[index+1][0] - y)**2 + (path[index+1][1] - x)**2) ** 0.5
        index_increase = 0
        for i in range(2,3):
            if index + i < len(path) - 1:
                next_d = ((path[index+i][0] - y)**2 + (path[index+i][1] - x)**2) ** 0.5
                if next_d < d:
                    index_increase = i
                    d = next_d
        index += index_increase
        
        # the cte is the estimate projected onto the normal of the path segment
        dist_delta = (dry * dx - drx * dy) / (dx * dx + dy * dy)    
        robot_orient = robot.orientation
        try:
            # path_orient = (m.atan2((path[index+2][0] - y),(path[index+2][1] - x)) + 2*m.pi) % (2*m.pi)
            path_orient = (m.atan2(path[index+1][0] - path[index][0], path[index+1][1] - path[index][1]) + 2*m.pi) % (2*m.pi)
            
        except:
            pass
            path_orient = (m.atan2((path[len(path)-1][0] - y),(path[len(path)-1][1] - x)) + 2*m.pi) % (2*m.pi)
        
        orient_diff = ((robot_orient - path_orient) + m.pi) % (2*m.pi) - m.pi
        steer = control.step(dist_delta)
        robot = robot.move(Map.grid, steer, velocity)
        # if abs(orient_diff) < 135 * m.pi / 180:
        #     steer = control.step(dist_delta)
        #     robot = robot.move(Map.grid, steer, dt * velocity)
        # else:
        #     steer = control.step(-dist_delta)
        #     robot = robot.move(Map.grid, steer, -dt * velocity)
        
        err += (dist_delta ** 2 + (10*orient_diff) ** 2)
        
        if show == True and N % steps == 0:
            Map.update_screen(fig, ax, robot, plan)
            ax.scatter(path[index+1][1],path[index+1][0], c = "red", s = 50)
            plt.pause(0.000001)
        
        N += 1

    return np.array(trajectory), err / n   

def twiddle(argrobot, tol=0.00001): 
    # Don't forget to call `make_robot` before you call `run`!
    p = [0.0, 0.0, 0.0]
    dp = [1.0, 1.0, 1.0]
    robot = deepcopy(argrobot)
    
    n = 850
    trajectory, best_err = run(robot, p, False, n)

    it = 0
    fig, ax = plt.subplots(1, 1, figsize =(18,8))
    
    while sum(dp) > tol:
        # print("Iteration {}, best error = {}".format(it, best_err))
        for i in range(len(p)):
            p[i] += dp[i]
            trajectory, err = run(robot, p, False, n)

            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] -= 2 * dp[i]
                trajectory, err = run(robot, p, False, n)

                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] += dp[i]
                    dp[i] *= 0.9
        
        it += 1
        
        if it % 20 == 0:
            trajectory, _ = run(robot, p, True, n)
            
            ax.cla()
            ax.plot(trajectory[:,1], trajectory[:,0])
            ax.plot(trajectory[:,3], trajectory[:,2])
            plt.pause(0.01)
            print("should show plots")
            
    
    
    return p, best_err

#Robot intial parameters
x = 200
y = 150
orientation = (m.pi * 45 / 180)

# Robot setup
width = 10
length = 20
steering = 0

#Movement parameters
velocity = 1.5                #Robot velocity = 5 cm/s
steps = 3
time_scale = 1.0

scale = [10, 10]

goal = [1200, 200]

fig, ax = plt.subplots(1, 1, figsize =(18,8))


Map = env.enviroment('maps/RobotMap.jpg', scale)

robot = Robots.steering_car(length, width, Map.pix_density)
robot.set(x/scale[1], y/scale[0], orientation)

plan = Plans.plan_with_reverse(Map.grid, [goal[0]/scale[0], goal[1]/scale[1]], 2)
path = plan.compute_path(Map, robot, fig, ax)

# plan.path = np.array(path)
# plan.path = plan.increase_resolution(2)

plan.smooth(0.3, 0.3)
path = plan.spath


# p, _ = twiddle(robot, 0.01)
p = [5.784263853039722, -0.04239115827521622, 14.551456094737718]

control = PID_Control(p[0], p[1], p[2])
    
N = 0
n = 700
index = 0
err = 0
trajectory = []
while not robot.check_goal([goal[0]/scale[0], goal[1]/scale[1]]) and N < n:
    y = robot.y
    x = robot.x
    robot_orient = robot.orientation
    
    trajectory.append([y, x, path[index+1][0], path[index+1][1]])
    
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
        N = n
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
    # path_orient = (m.atan2((path[index+1][0] - y),(path[index+2][1] - 1)) + 2*m.pi) % (2*m.pi)
    path_orient = (m.atan2(path[index+1][0] - path[index][0], path[index+1][1] - path[index][1]) + 2*m.pi) % (2*m.pi)

    
    orient_diff = ((robot_orient - path_orient) + m.pi) % (2*m.pi) - m.pi
    if abs(orient_diff) < 170 * m.pi / 180:
        steer = control.step(dist_delta)
        robot = robot.move(Map.grid, steer, velocity)
    else:
        steer = control.step(-dist_delta)
        robot = robot.move(Map.grid, steer, -velocity)
    
    Map.update_screen(fig, ax, robot, plan)
    ax.scatter(path[index+1][1],path[index+1][0], c = "red", s = 50)
    plt.pause(0.1 /(2 * time_scale))
    
    N += 1
