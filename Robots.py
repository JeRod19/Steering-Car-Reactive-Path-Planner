import math as m
import numpy as np
import random
 
class steering_car:

    def __init__(self, length = 20, width = 10, pix_density = 6.51):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0              #Radians
        
        self.length = length * pix_density  #Centimeters
        self.width = width * pix_density    #Centimeters
        self.pix_density = pix_density      #Pixels per centimer
        
        self.steering_noise    = 0.0
        self.distance_noise    = 0.0
        self.measurement_noise = 0.0
        
        self.num_collisions    = 0
        self.num_steps         = 0            
        
        self.camara = np.array([0,  self.length // 2])
        
        angle = 120
        view_angle = m.pi * angle / 180.0
        view_dist = 50 * pix_density 
        view_dist = self.length * 5
        
        self.horizon = []
        for ang in np.linspace(view_angle/2, -view_angle/2, angle):
            self.horizon.append([np.sin(ang) * view_dist, np.cos(ang) * view_dist])
        self.horizon = np.array(self.horizon)

    def set(self, new_x, new_y, new_orientation):

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation) % (2.0 * m.pi)

    def set_noise(self, new_s_noise, new_d_noise, new_m_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.steering_noise     = float(new_s_noise)
        self.distance_noise    = float(new_d_noise)
        self.measurement_noise = float(new_m_noise)
    
    def check_collision(self, realGrid):
        # Define robot´s corners
        corners = np.array([[-self.width/2, self.length / 2],   #front right
                            [self.width/2, self.length / 2],    #front left
                            [self.width/2, -self.length / 2],   #back left
                            [-self.width/2, -self.length / 2]]) #back right
        
        # Rotation matriz based on orientarion
        th = self.orientation
        rotMat = np.array([[m.cos(th), -m.sin(th)],[m.sin(th), m.cos(th)]])
        
        # Compute real corners position
        corners = np.matmul(corners, rotMat)
        corners += [self.y, self.x]
        
        # Ray vector 
        ray = corners[1,:] - corners[2,:]
        ray_size = np.linalg.norm(ray)
        ray /= ray_size
        
        # Robot´s back line
        back_size = np.linalg.norm(corners[2,:] - corners[3,:])
        back_line = np.linspace(corners[2,:], corners[3,:], int(back_size))
        
        for t in range(int(ray_size)):
            back_line += ray
            
            for [y, x] in back_line:
                x = int(x)
                y = int(y)
                if y < 0 or y >= len(realGrid) or \
                    x < 0 or x >= len(realGrid[0]) or \
                        realGrid[y][x] > 0 :
                    self.num_collisions += 1
                    return False

        return True        
        
    def check_goal(self, goal, threshold = 5.0):
        dist =  m.sqrt((float(goal[1]) - self.x) ** 2 + (float(goal[0]) - self.y) ** 2)
        return dist < threshold

    def move(self, realGrid, steering, distance, tolerance = 0.00001, max_steering_angle = m.pi / 4.0):
        
        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        
        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise) * self.pix_density

        # Execute motion
        turn = m.tan(steering2) * distance2 / self.length

        # make a new copy
        res = steering_car()
        res.length            = self.length
        res.width             = self.width
        res.pix_density       = self.pix_density
        
        res.steering_noise    = self.steering_noise
        res.distance_noise    = self.distance_noise
        res.measurement_noise = self.measurement_noise
        
        res.num_collisions    = self.num_collisions
        res.num_steps         = self.num_steps + 1
        res.camara            = self.camara
        res.horizon           = self.horizon

        steps = 5
        for i, orient in enumerate(np.linspace(self.orientation, self.orientation + turn, steps) % (2.0 * m.pi)):
            res.orientation = orient
            i += 1

            if abs(turn) < tolerance:
                # approximate by straight line motion              
                res.x = self.x + (distance2 * m.cos(self.orientation)) * (i/steps)
                res.y = self.y + (distance2 * m.sin(self.orientation)) * (i/steps)
    
            else:
                # approximate bicycle model for motion
                radius = distance2 / turn
                cx = self.x - (m.sin(self.orientation) * radius)
                cy = self.y + (m.cos(self.orientation) * radius)
                
                res.x = cx + (m.sin(res.orientation) * radius)
                res.y = cy - (m.cos(res.orientation) * radius)

            # If there is a collision, move to final position
            if res.check_collision(realGrid) == False:
                res.orientation = (self.orientation + turn) % (2.0 * m.pi)
                
                if abs(turn) < tolerance:
                    # approximate by straight line motion              
                    res.x = self.x + (distance2 * m.cos(self.orientation))
                    res.y = self.y + (distance2 * m.sin(self.orientation))
                else:
                    # approximate bicycle model for motion
                    radius = distance2 / turn
                    cx = self.x - (m.sin(self.orientation) * radius)
                    cy = self.y + (m.cos(self.orientation) * radius)
                    
                    res.x = cx + (m.sin(res.orientation) * radius)
                    res.y = cy - (m.cos(res.orientation) * radius)
                return res

        return res

    def sense(self):

        return [random.gauss(self.x, self.measurement_noise),
                random.gauss(self.y, self.measurement_noise)]

    def measurement_prob(self, measurement):

        # compute errors
        error_x = measurement[0] - self.x
        error_y = measurement[1] - self.y

        # calculate Gaussian 
        error = m.exp(- (error_x ** 2) / (self.measurement_noise ** 2) / 2.0) \
            / m.sqrt(2.0 * m.pi * (self.measurement_noise ** 2))
        error *= m.exp(- (error_y ** 2) / (self.measurement_noise ** 2) / 2.0) \
            / m.sqrt(2.0 * m.pi * (self.measurement_noise ** 2))

        return error
    
    def view(self, realGrid, senseGrid):
        rotMat = np.array([[m.cos(self.orientation), -m.sin(self.orientation)],
                           [m.sin(self.orientation), m.cos(self.orientation)]])
        
        camara = np.matmul(self.camara, rotMat)
        camara += [self.y, self.x]
        
        horizon = np.matmul(self.horizon, rotMat)
        horizon += camara
        
        something_new = False
        for p in horizon:
            ray = p - camara
            ray_size = np.linalg.norm(ray)
            ray /= ray_size
            ray_depth = 0
            
            for t in range(int(np.ceil(ray_size))):
                Rp = camara + ray*t
                Ry = int(Rp[0])
                Rx = int(Rp[1])
                
                #Delete old object
                if Ry > 0 and Ry < len(realGrid) and \
                    Rx > 0 and Rx < len(realGrid[0]) and \
                    realGrid[Ry][Rx] == 0 and senseGrid[Ry][Rx] == 1:
                    senseGrid[Ry][Rx] = 0
                    something_new = True       
                
                # Saved new detected object
                if Ry > 0 and Ry < len(realGrid) and \
                    Rx > 0 and Rx < len(realGrid[0]) and \
                    realGrid[Ry][Rx] == 1:
                    if  senseGrid[Ry][Rx] == 0:
                        something_new = True
                    senseGrid[Ry][Rx] = 1
                    ray_depth += 1
                    if ray_depth > 0:
                        break
                         
        
        return something_new

    def __repr__(self):
        # return '[x=%.5f y=%.5f orient=%.5f]'  % (self.x, self.y, self.orientation)
        return '[%.5f, %.5f]'  % (self.x, self.y)
