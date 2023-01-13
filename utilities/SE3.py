import numpy as np
from SO3 import SO3

'''
SE(3) class
Attributes: position, SO(3)
Methods: multiply
Methods: inverse
According to S.M.La Valle: Planning Algorithms, the most general form
of a matrix in SE(3) is given by (4.16) in which R \in SO(3) and v \in R^3.
C = R^3 x RP^3
'''

class SE3:
    # Constructor
    def __init__(self, position = np.array([]), orientation = '', T = np.array([]), custom = {}):
        if position.size != 0 and isinstance(orientation, SO3) != 0: # valid
            # given positional vector and paramterized vector of SO3
            self.p = position       # positional vector
            self.q = orientation.q  # quaternion
            self.R = orientation.R  # Rotation matrix 2D numpy array
            self.ypr = orientation.ypr # yaw pitch roll
            self.T = np.append(self.R, self.p.reshape(3,1), axis = 1)
            self.T = np.append(self.T, [[0, 0, 0, 1]], axis = 0)
        if T.size != 0: # valid
            # given the homogeneous transformation
            self.T = T             # homogeneous transformation
            self.p = self.T[:3, 3] # positional vector
            orientation = SO3(R = self.T[:3, :3]) # SO3 instance
            self.q = orientation.q # quaternion
            self.R = orientation.R # rotational matrix
            self.ypr = orientation.ypr #yaw pitch roll
        if position.size != 0 and isinstance(orientation, SO3) == False: # identity
            self.p = position      # positional vector
            self.q = SO3().q       # quaternion
            self.R = SO3().R       # rotational matrix
            self.ypr = SO3().ypr   # yaw pitch roll
            self.T = np.append(self.R, self.p.reshape(3,1), axis = 1)
            self.T = np.append(self.T, [[0, 0, 0, 1]], axis = 0)
        if bool(custom) == True:
            # build of SE3 instance given positional vector and customed rotational orders
            orders          = custom["composed_rotational_components"]["orders"]
            angles          = custom["composed_rotational_components"]["angles"]
            rotation_matrix = np.eye(3)
            for i in range(len(orders)):
                if orders[i] == "x":
                    temp = SO3(ypr = np.array([0, 0, angles[i]])).R
                if orders[i] == "y":
                    temp = SO3(ypr = np.array([0, angles[i], 0])).R
                if orders[i] == "z":
                    temp = SO3(ypr = np.array([angles[i], 0, 0])).R
                rotation_matrix = rotation_matrix.dot(temp)
            self.p      = custom["t"]
            self.R      = rotation_matrix
            orientation = SO3(R = rotation_matrix) # SO3 instance
            self.q      = orientation.q
            self.ypr    = orientation.ypr
            self.T = np.append(self.R, self.p.reshape(3,1), axis = 1)
            self.T = np.append(self.T, [[0, 0, 0, 1]], axis = 0)

    def multiply(self, SE3_new): # valid
        # binary operation
        # (2.24) Peter Corke
        T_composition = np.append(self.R.dot(SE3_new.R), self.p.reshape(3,1) + self.R.dot(SE3_new.p.reshape(3,1)), axis = 1)
        T_composition = np.append(T_composition, [[0, 0, 0, 1]], axis = 0)
        SE3_composition = SE3(T = T_composition)
        return SE3_composition


    def inverse(self):  #valid
        # (2.25) Peter Corke and (2.45) Siciliano are conformed.
        T_inverse = np.append(self.R.T, -self.R.T.dot(self.p.reshape(3,1)), axis = 1)
        T_inverse = np.append(T_inverse, [[0, 0, 0, 1]], axis = 0)
        SE3_inverse = SE3(T = T_inverse)
        # Implementation of vector-quaternion pair Peter Corke page 47
        # skip ( to be continued)

        return SE3_inverse


if __name__ == "__main__":
    # self-test code
    T_1 = np.array([
    [0.3536, -0.3536, 0.8660, 1.0000],
    [0.8839, -0.1768,-0.4330, 2.0000],
    [0.3062,  0.9186, 0.2500, 3.0000],
    [  0   ,    0   ,   0   , 1.0000]])
    SE3_instance_1 = SE3(T = T_1)

    T_2 = np.array([
    [0.4330, -0.7500, 0.5000, 4.0000],
    [0.7891,  0.0474,-0.6124, 2.0000],
    [0.4356,  0.6597, 0.6124, 5.0000],
    [  0   ,    0   ,   0   , 1.0000]])
    SE3_instance_2 = SE3(T = T_2)
    SE3_composition = SE3_instance_1.multiply(SE3_instance_2)
    print(SE3_composition.T)

    SE3_inverse = SE3_instance_2.inverse()
    print(SE3_inverse.T)
    SE3_identity = SE3(p = np.array([1, 2, 3]))
    SE3_identity_inverse = SE3_identity.inverse()
    print(SE3_identity_inverse.T)
