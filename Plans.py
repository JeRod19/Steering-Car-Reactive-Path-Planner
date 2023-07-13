import cv2 
import numpy as np
import math as m
from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter



def Skernel(size):
    if size % 2 == 0:
        return None
    c = size//2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if abs(c-i) > abs(c-j):
                if i % 2 == 0:
                    v = 1 + c - abs(c-i)
                else:
                    v = 0
            else:
                if j % 2 == 0:
                    v = 1 + c - abs(c-j)
                else:
                    v = 0
            kernel[i, j] = v
                
    return kernel / np.sum(kernel)

def Kernel(size):
    if size % 2 == 0:
        return None
    c = size//2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if abs(c-i) > abs(c-j):
                v = 1 + c - abs(c-i)
            else:
                v = 1 + c - abs(c-j)
            kernel[i, j] = v
                
    return kernel / np.sum(kernel)


class plan:

    def __init__(self, grid, goal, cost = 1):
        self.cost = cost
        self.goal = goal
        self.make_heuristic(grid, goal)
        self.path = []
        self.spath = []
        

    def make_heuristic(self, grid, goal):
        self.heuristic = np.zeros(grid.shape)
        
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                    self.heuristic[row,col] = m.sqrt(m.pow(goal[0]-row,2) + m.pow(goal[1]-col,2))
        
        ##self.heuristic /= self.heuristic.max()
        self.heuristic

    def quick_search_heuristic(self, grid, Arobot, goal):
        heuristic = np.zeros(grid.shape)
        
        finish = False
        width = int(Arobot.width * 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width))####
        freeGrid = cv2.filter2D(grid,-1,kernel)
        
        delta = [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0]
            ]
        
        nodes_list = [[0, goal[0], goal[1]]]
        
        while not finish:
            if len(nodes_list) == 0:
                finish = True
                print('###### Search terminated without success')
                return
                
            else:
                
                # remove node from list
                nodes_list.sort()
                nodes_list.reverse()
                node = nodes_list.pop()
                
                cost, y, x = node
                
                for move in delta:
                    x2 = x + delta[1]
                    y2 = y + delta[0]
                    
                    if y2 >= 0 and y2 < len(self.grid) and x2 >= 0 and x2 < len(self.grid[0]):
                        if freeGrid[y2, x2] == 0:
                            nodes_list.append([cost+1, y2, x2])
                        
                        
                
                    
                    
                    
                
            
            
            
        
        
        
        
        


    def compute_path(self, senseMap, Arobot, fig = None, ax = None):
        
        if self.heuristic == []:
            print("Heuristic must be defined to run A*")
            return 0
        
        robot = deepcopy(Arobot)
        robot.set_noise(0,0,0)
        self.grid = senseMap.grid
        mask = 1 - self.grid
        
        # radiation = cv2.GaussianBlur(self.grid, (15, 15), 15) * 1.3
        # radiation = cv2.GaussianBlur(radiation, (15, 15), 15)

        # radiation *= mask * self.heuristic.max()
        
        k = Skernel(13)
        radiation = cv2.filter2D(self.grid*1.5, ddepth=-1, kernel=k)
        
        radiation = radiation * mask + self.grid*1.5
        radiation = cv2.filter2D(src=radiation, ddepth=-1, kernel=k)
        
        k = Kernel(7)
        radiation = cv2.filter2D(src=radiation, ddepth=-1, kernel=k) * self.heuristic.max()
        
        self.H = self.heuristic * mask 
        
        p = 2 #.4
        self.H += p*radiation 


        
        if not fig == None:
            writer = PillowWriter(fps = 20, bitrate = 640000)
            writer.setup(fig, r'C:\Users\JRodr\Automated_system\Virtual_Robot\search_path.gif', dpi=100)
        
        node = 0
        robotDist = 10
        deg2rad = m.pi/180
        
        orient_range = 30 * deg2rad
        orient_segments = int(2 * m.pi / orient_range)

        # internal steering possibilities
        delta = np.linspace(-m.pi/4, m.pi/4, 9)
        delta = np.sort(abs(delta))
        for i in range(2, len(delta), 2):
            delta[i] *= -1
            
        
        states_size = (self.grid.shape[0], self.grid.shape[1], int(2 * m.pi / orient_range))
        self.closed = np.zeros(states_size)
        action = np.zeros(states_size)

        init = [robot.y, robot.x, robot.orientation]
        y = init[0]
        x = init[1]
        orient = init[2] #int((init[2] + (orient_range / 2)) // orient_range)
        h = self.H[int(y),int(x)]
        g = 0
        f = g + h

        open = [[f, g, h, y, x, orient]]

        found  = False # flag that is set when search complete
        resign = False # flag set if we can't find expand
        count  = 0
        
        self.closed[int(y),int(x), int(((orient + (orient_range / 2)) // orient_range) % orient_segments)] = 1        

        # with writer.saving(self.figure, r'C:\Users\JRodr\Automated_system\Virtual_Robot\search_path.gif', dpi=100):
        while not found and not resign:

            # check if we still have elements on the open list
            if len(open) == 0:
                resign = True
                print('###### Search terminated without success')
                return
                
            else:
                # remove node from list
                open.sort()
                open.reverse()
                next = open.pop()
                y = next[3]
                x = next[4]
                orient = next[5]
                g = next[1]
                node += 1

            # check if we are done

            if abs(y - self.goal[0]) < 3 and abs(x - self.goal[1]) < 3:
                found = True
                # print '###### A* search successful'

            else:
                
                # expand winning element and add to new open list
                for steering in delta:
                    robot.set(x, y, orient)
                    robot.num_collisions = 0
                    robot = robot.move(self.grid, steering, robotDist)
                    
                    y2 = robot.y
                    x2 = robot.x
                    orient2 = robot.orientation #int((robot.orientation + (orient_range / 2)) // orient_range)
                    colission = robot.num_collisions
                    
                    YBin = int(y2)
                    XBin = int(x2)
                    OrientBin = int(((orient2 + (orient_range / 2)) // orient_range) % orient_segments)

                    if y2 >= 0 and y2 < len(self.grid) and x2 >= 0 and x2 < len(self.grid[0]):
                        if self.closed[YBin, XBin, OrientBin] == 0 and self.grid[YBin, XBin] == 0 and colission == 0:
                            g2 = g + self.cost * 1 + self.closed[YBin-3:YBin+4, XBin-3:XBin+4, :].sum() * 0.6 #(1 + 2 *abs(steering))
                            h2 = self.H[int(y2), int(x2)]
                            f2 = g2 + h2
                            open.append([f2, g2, h2, y2, x2, orient2])
                            self.closed[YBin, XBin, OrientBin] = 1
                            action[YBin, XBin, OrientBin] = steering
                        else:
                            print(f"Close or collision: Node {node} cost{g} steer {steering} - > {YBin}, {XBin}, {OrientBin} collision = {colission}")
                    else:
                        print(f"Out of map: {y2}, {x2}, {orient2} collision = {colission}")
                            
            if count % 100 == 0 and not ax == None:
                ax.cla()
                expansion = self.closed.sum(2) / orient_segments
                expansion += self.grid
                ax.imshow(expansion, origin = 'lower')
                plt.pause(0.0001)
                writer.grab_frame()
            count += 1

        # extract the path
        
        if not ax == None:
            ax.cla()
            expansion = self.closed.sum(2) / orient_segments
            expansion += self.grid
            expansion[int(self.goal[0]), int(self.goal[1])] = 1
            ax.imshow(expansion, origin = 'lower')
            plt.pause(0.0001)
            writer.grab_frame()

        self.invpath = []
        self.invpath.append([y, x])
        robot.set(x, y, orient)
        while abs(y - init[0]) > 5 or abs(x - init[1]) > 5:
            YBin = int(y)
            XBin = int(x)
            OrientBin = int(((orient + (orient_range / 2)) // orient_range) % orient_segments)
            steering = action[YBin, XBin, OrientBin]
            robot = robot.move(self.grid, steering, -robotDist)
                    
            y = robot.y
            x = robot.x
            orient = robot.orientation

            self.invpath.append([y, x])

        self.path = []
        for i in range(len(self.invpath)):
            self.path.append(self.invpath[len(self.invpath) - 1 - i])
                        
        self.path = np.array(self.path)
        
        if not ax == None:
            ax.plot(self.path[:,1], self.path[:,0])
            writer.grab_frame()
            writer.finish()
        
        
        n = len(self.path)
        kernel = np.ones((7,1)) / 7
        self.path[1:n-1,:] = cv2.filter2D(self.path[1:n-1,:], -1, kernel)
        if not ax == None:
            ax.plot(self.path[:,1], self.path[:,0])
            plt.pause(0.0001)

        return self.smooth(0.1, 0.3)
        


    def smooth(self, weight_data = 0.1, weight_smooth = 0.1, 
                tolerance = 0.000001):
    
    
        self.spath = self.path.copy()
    
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(self.path)-1):
                for j in range(len(self.path[0])):
                    aux = self.spath[i,j]
                    
                    self.spath[i,j] += weight_data * (self.path[i,j] - self.spath[i,j])
                    
                    self.spath[i,j] += weight_smooth * (self.spath[i-1,j] + self.spath[i+1,j] - (2.0 * self.spath[i,j]))
                    
                    if i >= 2:
                        self.spath[i,j] += 0.5 * weight_smooth * (2 * self.spath[i-1,j] - self.spath[i-2,j] - self.spath[i,j])
                    if i <= len(self.path) - 3:
                        self.spath[i,j] += 0.5 * weight_smooth * (2 * self.spath[i+1,j] - self.spath[i+2,j] - self.spath[i,j])
                
                    change += abs(aux - self.spath[i,j])
        
        return self.spath
                


class plan_with_reverse:

    def __init__(self, grid, goal, cost = 1):
        self.cost = cost
        self.goal = goal
        self.make_heuristic(grid, goal)
        self.path = []
        self.spath = []
        

    def make_heuristic(self, grid, goal):
        self.heuristic = np.zeros(grid.shape)
        
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                    self.heuristic[row,col] = m.sqrt(m.pow(goal[0]-row,2) + m.pow(goal[1]-col,2))
        
        ##self.heuristic /= self.heuristic.max()
        self.heuristic *= 1.2


    def compute_path(self, senseMap, Arobot, fig = None, ax = None):
        
        if self.heuristic == []:
            print("Heuristic must be defined to run A*")
            return 0
        
        robot = deepcopy(Arobot)
        robot.set_noise(0,0,0)
        self.grid = senseMap.grid
        mask = 1 - self.grid
        
        # radiation = cv2.GaussianBlur(self.grid, (15, 15), 15) * 1.3
        # radiation = cv2.GaussianBlur(radiation, (15, 15), 15)

        # radiation *= mask * self.heuristic.max()
        
        k = Skernel(13)
        radiation = cv2.filter2D(self.grid*1.5, ddepth=-1, kernel=k)
        
        radiation = radiation * mask + self.grid*1.5
        radiation = cv2.filter2D(src=radiation, ddepth=-1, kernel=k)
        
        k = Kernel(7)
        radiation = cv2.filter2D(src=radiation, ddepth=-1, kernel=k) * self.heuristic.max()
        
        self.heuristic *= mask    
        self.radiation = radiation * mask


        
        if not fig == None:
            writer = PillowWriter(fps = 20, bitrate = 640000)
            writer.setup(fig, r'C:\Users\JRodr\Automated_system\Virtual_Robot\search_path.gif', dpi=100)
        
        node = 0
        robotDist = robot.length * 1.5
        deg2rad = m.pi/180
        
        orient_range = 20 * deg2rad
        orient_segments = int(2 * m.pi / orient_range)

        # internal steering possibilities
        delta = np.linspace(-m.pi/4, m.pi/4, 9)
        delta = np.sort(abs(delta))
        
        directions = [1, 1]
        
        for i in range(2, len(delta), 2):
            delta[i] *= -1
            
        
        states_size = (self.grid.shape[0], self.grid.shape[1], orient_segments, len(directions))
        self.closed = np.zeros(states_size[0:3])
        action = np.zeros(states_size)

        init = [robot.y, robot.x, robot.orientation]
        y = init[0]
        x = init[1]
        orient = init[2] #int((init[2] + (orient_range / 2)) // orient_range)
        h = self.heuristic[int(y),int(x)]
        g = 0
        f = g + h

        open = [[f, g, h, y, x, orient]]

        found  = False # flag that is set when search complete
        resign = False # flag set if we can't find expand
        count  = 0
        
        self.closed[int(y),int(x), int(((orient + (orient_range / 2)) // orient_range) % orient_segments)] = 1        

        # with writer.saving(self.figure, r'C:\Users\JRodr\Automated_system\Virtual_Robot\search_path.gif', dpi=100):
        while not found and not resign:

            # check if we still have elements on the open list
            if len(open) == 0:
                resign = True
                print('###### Search terminated without success')
                return
                
            else:
                # remove node from list
                open.sort()
                open.reverse()
                next = open.pop()
                y = next[3]
                x = next[4]
                orient = next[5]
                g = next[1]
                node += 1
            
            # node_proggress = self.closed[int(y),int(x), :].sum()
            # if node_proggress > orient_segments * 0.85:
            #     continue
            # a = 0
            # if y < 10 and x < 10:
            #     a = 1
            #     continue
            size = 5
            yd = int(y)-size
            if yd < 0: yd = 0
            
            yu = int(y)+size+1
            if yu > len(self.grid): yu = len(self.grid)
            
            xd = int(x)-size
            if xd < 0: xd = 0
            
            xu = int(x)+size+1
            if xu > len(self.grid[0]): xu = len(self.grid[0])

            
            zone_proggress = self.closed[yd:yu, xd:xu, :].sum()
            if zone_proggress > orient_segments * 4 * size * size * 0.1:
                continue
            
            
            

            # check if we are done

            if abs(y - self.goal[0]) < 10 and abs(x - self.goal[1]) < 10:
                found = True
                # print '###### A* search successful'

            else:
                
                # expand winning element and add to new open list
                
                for direct in directions:
                    over_explore_cost = None
                    for steering in delta:
                        robot.set(x, y, orient)
                        robot.num_collisions = 0
                        robot = robot.move(self.grid, steering, robotDist * direct)
                        
                        y2 = robot.y
                        x2 = robot.x
                        orient2 = robot.orientation #int((robot.orientation + (orient_range / 2)) // orient_range)
                        colission = robot.num_collisions
                        
                        YBin = int(y2)
                        XBin = int(x2)
                        OrientBin = int(((orient2 + (orient_range / 2)) // orient_range) % orient_segments)
    
                        if y2 >= 0 and y2 < len(self.grid) and x2 >= 0 and x2 < len(self.grid[0]):
                            if self.closed[YBin, XBin, OrientBin] == 0 and self.grid[YBin, XBin] == 0 and colission == 0:
                                if over_explore_cost == None:
                                    over_explore_cost = self.closed[YBin-2:YBin+3, XBin-2:XBin+3, :].sum()
                                    
                                reverse_cost =  1 + 5 * (1 - direct)  
                                steering_cost =  1 * abs(steering)
                                g2 = g + (self.cost + 6 * over_explore_cost + steering_cost * 2) * reverse_cost
                            
                                h2 = 3 *( 0.6*self.heuristic[YBin, XBin] + 1.3*self.radiation[YBin, XBin])
                                
                                f2 = g2 + h2
                                open.append([f2, g2, h2, y2, x2, orient2])
                                self.closed[YBin, XBin, OrientBin] = 1
                                action[YBin, XBin, OrientBin, :] = [steering, direct]
                            else:
                                print(f"Close or collision: Node {node} cost{g} steer {steering} - > {YBin}, {XBin}, {OrientBin} collision = {colission}")
                        else:
                            print(f"Out of map: {y2}, {x2}, {orient2} collision = {colission}")
                            
            if count % 50 == 0 and not ax == None:
                ax.cla()
                expansion = self.closed.sum(2) / orient_segments
                expansion += self.grid
                ax.imshow(expansion, origin = 'lower')
                plt.pause(0.00001)
                #writer.grab_frame()
            count += 1

        # extract the path
        
        if not ax == None:
            ax.cla()
            expansion = self.closed.sum(2) / orient_segments
            expansion += self.grid
            expansion[int(self.goal[0]), int(self.goal[1])] = 1
            ax.imshow(expansion, origin = 'lower')
            plt.pause(0.00001)
            #writer.grab_frame()

        self.invpath = []
        self.invpath.append([y, x])
        robot.set(x, y, orient)
        while abs(y - init[0]) > 10 or abs(x - init[1]) > 10:
            YBin = int(y)
            XBin = int(x)
            OrientBin = int(((orient + (orient_range / 2)) // orient_range) % orient_segments)
            steering, direct = action[YBin, XBin, OrientBin, :]
            robot = robot.move(self.grid, steering, -robotDist * direct)
                    
            y = robot.y
            x = robot.x
            orient = robot.orientation

            self.invpath.append([y, x])

        self.invpath.append([init[0], init[1]])
        self.path = []
        for i in range(len(self.invpath)):
            self.path.append(self.invpath[len(self.invpath) - 1 - i])
                        
        self.path.append([self.goal[0], self.goal[1]])    
        self.path = np.array(self.path)
        
        if not ax == None:
            ax.plot(self.path[:,1], self.path[:,0])
            writer.grab_frame()
            writer.finish()
        
        
        # n = len(self.path)
        # kernel = np.ones((7,1)) / 7
        # self.path[1:n-1,:] = cv2.filter2D(self.path[1:n-1,:], -1, kernel)
        # if not ax == None:
        #     ax.plot(self.path[:,1], self.path[:,0])
        #     plt.pause(0.0001)
        
        self.path = self.increase_resolution(4)
        
        return self.path
        #return self.smooth(0.1, 0.3)
        

    def increase_resolution(self, scale = 3):
        new_number_points = (self.path.shape[0] - 1) * scale + 1
        
        new_path = np.zeros((new_number_points, 2))
        
        for i in range(self.path.shape[0] - 1):
            for j in range(scale):
                p = j / scale
                new_point = self.path[i] * (1 - p) + self.path[i+1] * p 
                new_path[i*scale + j] = new_point
        
        new_path[-1] = self.path[-1]
        
        return new_path
        
        
        
    def smooth(self, weight_data = 0.1, weight_smooth = 0.1, tolerance = 0.000001):
    
        self.spath = self.path.copy()
    
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(self.path)-1):
                for j in range(len(self.path[0])):
                    aux = self.spath[i,j]
                    
                    self.spath[i,j] += weight_data * (self.path[i,j] - self.spath[i,j])
                    
                    self.spath[i,j] += weight_smooth * (self.spath[i-1,j] + self.spath[i+1,j] - (2.0 * self.spath[i,j]))
                    
                    if i >= 2:
                        self.spath[i,j] += 0.5 * weight_smooth * (2 * self.spath[i-1,j] - self.spath[i-2,j] - self.spath[i,j])
                    if i <= len(self.path) - 3:
                        self.spath[i,j] += 0.5 * weight_smooth * (2 * self.spath[i+1,j] - self.spath[i+2,j] - self.spath[i,j])
                
                    change += abs(aux - self.spath[i,j])
        
        return self.spath

