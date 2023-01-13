import numpy as np
from scipy.linalg import logm, expm

'''
SO(3) class
Attributes: quaternion, rotation matrix, roll pitch yaw
Methods: rot_2_rpy, rot_2_quaternion, rpy_2_rot, quaternion_2_rot
Methods: inverse quaternion, quaternion multiplication
An SO(3) instance is defined using one of its attributes.
Given an SO(3) instance, it is natural to transform back and forth between representation.
page 141. Planning Algorithms. A group is a set together with binary operation.
'''

class SO3:
    # Constructor
    def __init__(self, q = np.array([]), R = np.array([]), ypr = np.array([]), axis_angle = np.array([]), custom = {}):
        if q.size != 0:
            self.q = q # quaternion
            self.R = self.quaternion_2_rot()
            self.ypr = self.rot_2_ypr()
        if R.size != 0:
            self.R = R # Rotation matrix
            self.q = self.rot_2_quaternion()
            self.ypr = self.rot_2_ypr()
        if ypr.size != 0:
            self.ypr = ypr # yaw pitch roll
            self.R = self.ypr_2_rot()
            self.q = self.rot_2_quaternion()
        if axis_angle.size != 0:
            self.axis_angle  = axis_angle
            self.R           = self.axis_angle_to_rot()
            self.q           = self.rot_2_quaternion()
            self.ypr         = self.rot_2_ypr()
        if q.size == 0 and R.size == 0 and ypr.size == 0 and axis_angle.size == 0: # identity
            self.q = np.array([1, 0, 0, 0])
            self.R = self.quaternion_2_rot()
            self.ypr = self.rot_2_ypr()
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
            self.R      = rotation_matrix
            orientation = SO3(R = rotation_matrix) # SO3 instance
            self.q      = orientation.q
            self.ypr    = orientation.ypr

    def multiply(self, SO3_new):
       # SO(3) binary operation
       SO3_composition = self.quaternion_multiply(SO3_new)
       return SO3_composition

    def inverse(self):
        SO3_inverse = self.quaternion_inv()
        return SO3_inverse

    def rot_2_quaternion(self):   # Valid
        # (2.34) (2.35) Siciliano not conform with (4.24) ~ (4.27) S.M. La Valle: Planning Algorithms
        q_1 = 0.5 * np.sqrt(abs(self.R[0,0] + self.R[1,1] + self.R[2,2] + 1))
        q_2 = 0.5 * np.sign(self.R[2,1] - self.R[1,2]) * np.sqrt(abs(self.R[0,0] - self.R[1,1] - self.R[2, 2] + 1))
        q_3 = 0.5 * np.sign(self.R[0,2] - self.R[2,0]) * np.sqrt(abs(self.R[1,1] - self.R[2,2] - self.R[0, 0] + 1))
        q_4 = 0.5 * np.sign(self.R[1,0] - self.R[0,1]) * np.sqrt(abs(self.R[2,2] - self.R[0,0] - self.R[1, 1] + 1))
        q = np.array([q_1, q_2, q_3, q_4])
        self.q = q
        return self.q

    def rot_2_ypr(self):   # Valid
        # (3.47) ~ (3.49) S.M. La Valle: Planning Algorithms
        # (2.22) Siciliano
        # Although Siciliano and La Valle have different notion for elementary rotation roll, pitch, and yaw
        # Siciliano: rotating about axis x (yaw) ,  rotating about axis y (pitch), rotating about axis z (roll)
        #  La Valle: rotating about axis x (roll),  rotating about axis y (pitch), rotating about axis z (yaw)
        # The order in which elementary rotation applied is the same: R = R_z * R_y * R_x
        # All elementary rotations are defined with respect to the world frame ( conformity between Siciliano and La Valle)
        alpha = np.arctan2(self.R[1][0],self.R[0][0]) # radian
        beta = np.arctan2(-self.R[2][0], np.sqrt(np.square(self.R[2][1]) + np.square(self.R[2][2]))) # radian
        gamma = np.arctan2(self.R[2][1], self.R[2][2])   # radian
        ypr = np.array([alpha, beta, gamma])
        self.ypr = ypr
        return self.ypr

    def quaternion_2_rot(self): # Valid
        R = np.zeros([3,3], dtype = np.float32)
        q_1 = self.q[0]
        q_2 = self.q[1]
        q_3 = self.q[2]
        q_4 = self.q[3]

        R[0,0] = 2 * (np.square(q_1) + np.square(q_2)) -1
        R[0,1] = 2 * (q_2 * q_3 - q_1 * q_4)
        R[0,2] = 2 * (q_2 * q_4 + q_1 * q_3)

        R[1,0] = 2 * (q_2 * q_3 + q_1 * q_4)
        R[1,1] = 2 * (np.square(q_1) + np.square(q_3)) -1
        R[1,2] = 2 * (q_3 * q_4 - q_1 * q_2)

        R[2,0] = 2 * (q_2 * q_4 - q_1 * q_3)
        R[2,1] = 2 * (q_3 * q_4 + q_1 * q_2)
        R[2,2] = 2 * (np.square(q_1) + np.square(q_4)) - 1
        self.R = R
        return self.R

    def axis_angle_to_rot(self):
        # Ref: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        # Caution: the formulation used to calculate Rotation Matrix to align Vector A to Vector B in 3d
        # differs from Peter Corke Equation 2.18 Chapter 2 - Representing Position and Orientation,
        # which is inverse problem of finding the rotation matrix between two coordinate frames from
        # the axis and angle representation. The axis-angle given in the current context is different, where
        # the angle is the angle between two vectors, and rotational axis is the cross product of two given vectors

        theta     = self.axis_angle[0]
        x         = self.axis_angle[1]
        y         = self.axis_angle[2]
        z         = self.axis_angle[3]

        skew_w    = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        R         = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(theta))

        return R

    def ypr_2_rot(self):   # Valid
        # (3.42) S. M. La Valle: Planning Algorithms
        alpha = self.ypr[0] # yaw
        beta = self.ypr[1]  # pitch
        gamma = self.ypr[2] # roll
        R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+ np.sin(alpha)*np.sin(gamma)],               [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)], [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
        self.R = R
        return self.R

    def rot_multiply(self, R):  # Valid
        R_new = self.R.dot(R)
        SO3_new = SO3(R = R_new)
        return SO3_new

    def rot_inv(self):   # Valid
        R_inv = np.linalg.inv(self.R)
        SO3_new = SO3(R = R_inv)
        return SO3_new

    def quaternion_multiply(self, SO3_new):   # Valid
        # (4.19) S. M. La Valle: Planning Algorithms
        # Conforming with (2.37) Siciliano
        q = SO3_new.q
        q1_new = self.q[0]*q[0] - self.q[1]*q[1] - self.q[2]*q[2] - self.q[3]*q[3]
        q2_new = self.q[0]*q[1] + self.q[1]*q[0] + self.q[2]*q[3] - self.q[3]*q[2]
        q3_new = self.q[0]*q[2] + self.q[2]*q[0] + self.q[3]*q[1] - self.q[1]*q[3]
        q4_new = self.q[0]*q[3] + self.q[3]*q[0] + self.q[1]*q[2] - self.q[2]*q[1]
        q_new = np.array([q1_new, q2_new, q3_new, q4_new])
        SO3_composition = SO3(q = q_new)
        return SO3_composition

    def quaternion_inv(self): # Valid
        # (2.36) Siciliano
        q1_inv =  self.q[0]
        q2_inv = -self.q[1]
        q3_inv = -self.q[2]
        q4_inv = -self.q[3]
        q_inv = np.array([q1_inv, q2_inv, q3_inv, q4_inv])
        SO3_inv = SO3(q = q_inv)
        return SO3_inv

if __name__ == "__main__":
    # self-test code
    R= np.array([[9.90515610e-01,  1.20701310e-01, -6.56507421e-02],
 [-1.20944339e-01,  9.92659252e-01,  2.74435784e-04],
 [ 6.52019413e-02,  7.66825267e-03,  9.97842625e-01]])
    R_1 = np.array([[0.7071, 0, 0.7071],
    [0.6124, 0.5000, -0.6124],
   [-0.3536, 0.8660, 0.3536]])
    SO3_1 = SO3(R = R)
    SO3_2 = SO3(R = R_1)
    SO3_inv = SO3_2.inverse()
    print(SO3_2.q)
    ans = SO3_1.multiply(SO3_2)
    print(ans.R)
    print(SO3_inv.R)
    identity = SO3()
