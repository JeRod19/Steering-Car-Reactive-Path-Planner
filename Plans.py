import cv2 
import numpy as np
import math as m

from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
from copy import deepcopy
import time
import threading


class nodeClass:
    def __init__(self, complexCost, movementCost, y, x, orient, direction, prevNode = None):
        self.complexCost = complexCost
        self.movementCost = movementCost
        self.y = y
        self.x = x
        self.orient = orient
        self.direction = direction
        self.prevNode = prevNode

        if prevNode == None:
            self.noReverseMovements = 0
        else:
            self.noReverseMovements = prevNode.noReverseMovements
    
    def __lt__(self, other):
        return self.complexCost < other.complexCost

class plan_with_reverse:

    def __init__(self, grid, goal, cost = 1, minWidth = 10):
        self.cost = cost
        self.goal = goal
        self.minWidth = minWidth
        self.grid = grid.copy()
        self.maxODGM_distance = 15
        self.compute_first_ODGM(grid)
        self.compute_first_GDM(grid, minWidth, goal)
        self.path = []
        self.spath = []
        

        if False:
            fig, ax = plt.subplots(2, 2)
            ax[0,0].imshow(grid, origin = 'lower')
            ax[0,1].imshow(self.maxODGM_distance - self.ODGM, cmap = 'gray', origin = 'lower')
            ax[1,0].imshow(self.GDM, cmap = 'gray', origin = 'lower', vmin = 0, vmax = 390)
            ax[1,0].scatter(goal[1], goal[0], c="red", s = 5)
            final = 1 + 0.5*((self.maxODGM_distance - self.ODGM) / self.maxODGM_distance)
            ax[1,1].imshow(self.GDM * final, cmap = 'gray', origin = 'lower', vmin = 0, vmax = 420) #vmax = np.unique(self.GDM)[-2]
            ax[1,1].scatter(goal[1], goal[0], c="red", s = 5)
            plt.show()
            plt.pause(0.001)
            pass
            # plt.close()
        
##########################################################
####################### Heuristics #######################
##########################################################

    def compute_first_ODGM(self, grid):
        self.ODGM = np.ones(grid.shape) * self.maxODGM_distance

        nodesToExpand = self.getNewObstacles(grid)
 
        closed_nodes = self.getClosedNodesGrid(nodesToExpand)

        while(len(nodesToExpand) > 0):
            node = nodesToExpand.pop(0)

            dist = np.sqrt( (node[1] - node[3][0])**2 + (node[2] - node[3][1])**2 )
            if self.updateCurrentODGMNode(nodeDist = dist, nodeRow = node[1], nodeCol = node[2]):
                [newNodes, closed_nodes] = self.addNewNodes(nodeToExpand = node, closed_nodes = closed_nodes)
                nodesToExpand += newNodes
                
        # self.maxODGM_distance = np.max(self.ODGM)

    def getNewObstacles(self, grid):
        newObstaclesNodes = []

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if grid[row, col] == 1:
                    origin = [row, col]
                    newObstaclesNodes.append([0, row, col, origin])

        return newObstaclesNodes

    def getClosedNodesGrid(self, nodesToExpand):
        closed_nodes = np.zeros((int(np.ceil(self.grid.shape[0])), int(np.ceil(self.grid.shape[1]))))

        for [_, row, col, _] in nodesToExpand:
            closed_nodes[row, col] = 1
        
        return closed_nodes

    def updateCurrentODGMNode(self, nodeDist, nodeRow, nodeCol):
        if nodeDist < self.ODGM[nodeRow, nodeCol]:
            self.ODGM[nodeRow, nodeCol] = nodeDist
            return True
        else:
            return False

    def nodeInsideGrid(self, row, col):
        if  row >= 0 and row < self.grid.shape[0] and col >= 0 and col < self.grid.shape[1]:
            return True
        else:
            return False
        
    def nodeIsNotObstacle(self, row, col):
        if  self.grid[row, col] == 0:
            return True
        else:
            return False

    def addNewNodes(self, nodeToExpand, closed_nodes):
        expandedNodes = []

        for [i, j] in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            row = nodeToExpand[1] + i
            col = nodeToExpand[2] + j
            dist = np.sqrt( (row - nodeToExpand[3][0])**2 + (col - nodeToExpand[3][1])**2 )


            if self.nodeInsideGrid(row, col) and closed_nodes[row, col] == 0 and dist < self.ODGM[row, col] and self.nodeIsNotObstacle(row, col):
                closed_nodes[row, col] = 1

                expandedNodes.append([dist, row, col, nodeToExpand[3]])
        
        return expandedNodes, closed_nodes             
            
    def update_ODGM(self, newGrid):
        nodesToExpand = self.getNewObstacles(newGrid - self.grid)
 
        closed_nodes = self.getClosedNodesGrid(nodesToExpand)

        while(len(nodesToExpand) > 0):
            node = nodesToExpand.pop(0)

            dist = np.sqrt( (node[1] - node[3][0])**2 + (node[2] - node[3][1])**2 )
            if self.updateCurrentODGMNode(nodeDist = dist, nodeRow = node[1], nodeCol = node[2]):
                [newNodes, closed_nodes] = self.addNewNodes(nodeToExpand = node, closed_nodes = closed_nodes)
                nodesToExpand += newNodes


    def compute_first_GDM(self, grid, winSize, goal):
        self.GDM = np.ones(grid.shape) * grid.size
        winSize = int(np.ceil(winSize / 2)) + 1 

        nodes = []
        nodes.append([0, int(goal[0]), int(goal[1])])
        closed_nodes = np.zeros(grid.shape)
        closed_nodes[int(goal[0]), int(goal[1])] = 1

        while(len(nodes) > 0):
            [d, row, col] = nodes.pop(0)

            if grid[max(0,row-winSize):min(row+winSize, grid.shape[0]), max(0,col-winSize):min(col+winSize, grid.shape[1])].sum() > 0:
                continue
            
            self.GDM[row, col] = d

            for [i, j] in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                row2 = row + i
                col2 = col + j
                
                if self.nodeInsideGrid(row2, col2) and closed_nodes[row2, col2] == 0 and grid[row2, col2] == 0:
                    nodes.append([d+1, row2, col2])
                    closed_nodes[row2, col2] = 1
        
        

    def update_heuristics(self, senseMap):

        self.update_ODGM(senseMap.grid)
        self.compute_first_GDM(senseMap.grid, self.minWidth, self.goal)

        if False:
            fig, ax = plt.subplots(2, 2)
            ax[0,0].imshow(senseMap.grid, origin = 'lower')
            ax[0,1].imshow(self.ODGM, cmap = 'gray', origin = 'lower')
            ax[1,0].imshow(self.GDM, cmap = 'gray', origin = 'lower')
            final = 1 + 0.5*((self.maxODGM_distance - self.ODGM) / self.maxODGM_distance)
            ax[1,1].imshow(self.GDM * final, cmap = 'gray', origin = 'lower')
            plt.show()
            plt.pause(0.0001)
            # plt.close()

        #update2

        self.grid = senseMap.grid

##########################################################
##################### Path planning ######################
##########################################################

    def expandNode(self, openNodes, robot, steerings, directions):
        # Sort and take node to expand
        if len(openNodes) == 0:
            return None
        
        openNodes.sort()
        currNode = openNodes.pop(0)
    
        # Get node attributes
        g = currNode.movementCost
        y = currNode.y
        x = currNode.x
        orient = currNode.orient
        direct = currNode.direction

         # check if we are done
        if abs(y - self.goal[0]) <= 3 and abs(x - self.goal[1]) <= 3:
            global finishNode
            finishNode = currNode


        robotDist = robot.length * 0.8
        orient_segments = self.closed.shape[2]
        orient_range = 2 * m.pi / orient_segments

        # Check if zone is over explored to skip
        size = 3

        yd = max(int(y)-size, 0)
        yu = min(int(y)+size+1, len(self.grid))
        xd = max(int(x)-size, 0)
        xu = min(int(x)+size+1, len(self.grid[0]))

        zone_proggress = self.closed[yd:yu, xd:xu, :].sum()
        if zone_proggress > orient_segments * size * size * 0.8:
            return None
        
        for direct in directions:
            for steer in steerings:
                robot.set(x, y, orient)
                robot.num_collisions = 0
                robot = robot.move(self.grid, steer, robotDist * direct)

                y2 = robot.y
                x2 = robot.x
                orient2 = robot.orientation 
                colission = robot.num_collisions

                YBin = int(y2)
                XBin = int(x2)
                OrientBin = int(((orient2 + (orient_range / 2)) // orient_range) % orient_segments)
                
                if self.nodeInsideGrid(y2, x2):
                    if self.closed[YBin, XBin, OrientBin] == 0 and self.grid[YBin, XBin] == 0 and colission == 0:
                        reverse_cost =  1 + 5 * (1 - direct) * currNode.noReverseMovements
                        steering_cost = abs(steer)

                        g2 = g + (self.cost + steering_cost * 2) * reverse_cost 
                    
                        h2 = self.heuristic[YBin, XBin]
                        
                        f2 = g2 + h2

                        auxNode = nodeClass(
                            complexCost = f2,
                            movementCost = g2, 
                            y = y2, 
                            x = x2, 
                            orient = orient2, 
                            direction = direct,
                            prevNode = currNode
                        )

                        if direct == -1:
                            auxNode.noReverseMovements += 1

                        openNodes.append(auxNode)
                        self.closed[YBin, XBin, OrientBin] = 1
        return None    

    def compute_path(self, senseMap, Arobot, fig = None, ax = None):
        start = time.time()
        robot = deepcopy(Arobot)
        robot.set_noise(0,0,0)
        self.grid = senseMap.grid.copy()
        
        self.heuristic = (1 + 0.25*((self.maxODGM_distance - self.ODGM) / self.maxODGM_distance)) * self.GDM ** 1
        
        # possible porientations for robot
        orient_range = 10 * m.pi / 180
        orient_segments = int(2 * m.pi / orient_range)

        # internal steering possibilities
        steerings = np.linspace(-m.pi/4, m.pi/4, 9)
        steerings = np.sort(abs(steerings)) 
        for i in range(2, len(steerings), 2):
            steerings[i] *= -1

        directions = [1, -1]
        
        y = robot.y
        x = robot.x
        orient = robot.orientation
        h = self.heuristic[int(y),int(x)]
        g = 0
        f = g + h

        openNodes = []
        openNodes.append(
            nodeClass(
            complexCost = f,
            movementCost = g, 
            y = y, 
            x = x, 
            orient = orient, 
            direction = 0
            )
        )

        found  = False # flag that is set when search complete
        resign = False # flag set if we can't find expand
        count  = 0
        self.closed = np.zeros((self.grid.shape[0], self.grid.shape[1], orient_segments))
        self.closed[int(y),int(x), int(((orient + (orient_range / 2)) // orient_range) % orient_segments)] = 1        

        global finishNode
        finishNode = None
        threadsList = list()

        while not found and not resign:

            # check if we still have elements on the open list
            if len(openNodes) == 0 :
                if  len(threading.enumerate()) == 0:
                    resign = True
                    print('###### Search terminated without success')
                    return
                else:
                    print(f"Threads remain: {len(threading.enumerate())}")
            else:
                xThread = threading.Thread(target= self.expandNode, args=(openNodes, deepcopy(Arobot), steerings, directions))
                threadsList.append(xThread)
                xThread.start()

            if finishNode != None:
                winnerNode = finishNode
                found = True
                break
                            
            count += 1

        self.path = [[winnerNode.y, winnerNode.x, winnerNode.direction]]

        while winnerNode.prevNode != None:
            winnerNode = winnerNode.prevNode
            self.path.append([winnerNode.y, winnerNode.x, winnerNode.direction])

        self.path.reverse()
        self.path.pop()
        self.path.append([self.goal[0], self.goal[1], 1])    

        self.path = np.array(self.path)
       
        self.path = self.increase_resolution(2)

        return self.path
    
    def increase_resolution(self, scale = 3):
        new_number_points = (self.path.shape[0] - 1) * scale + 1
        
        new_path = np.zeros((new_number_points, 3))
        
        for i in range(self.path.shape[0] - 1):
            for j in range(scale):
                p = j / scale
                new_point = self.path[i] * (1 - p) + self.path[i+1] * p 
                new_path[i*scale + j] = new_point
        
        new_path[-1] = self.path[-1]
        
        return new_path  
        
