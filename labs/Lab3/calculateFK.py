import numpy as np
from math import pi
import math

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        self.x1 = 0.082
        self.x2 = 0.125
        self.x3 = 0.259
        self.x4 = 0.088

        self.y1 = 0.015
        
        self.z1 = 0.141
        self.z2 = 0.192
        self.z3 = 0.195
        self.z4 = 0.121
        self.z5 = 0.083
        self.z6 = -0.051
        self.z7 = -0.159
        

        pass

    def forward(self, q):

        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        # Define the base-to-joint transformation matrix for the first joint
        AW0 = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,self.z1],
                        [0,0,0,1]])
        # Define the transformation matrices for each joint in the robot
        A01 = np.array([[math.cos(q[0]), 0, math.sin(q[0]), 0], 
                        [math.sin(q[0]), 0, -math.cos(q[0]), 0], 
                        [0, 1, 0, self.z2], 
                        [0, 0, 0, 1]])
        A12 = np.array([[math.cos(-q[1]), 0, -math.sin(-q[1]), 0], 
                        [math.sin(-q[1]), 0, math.cos(-q[1]), 0], 
                        [0, -1, 0, 0], 
                        [0, 0, 0, 1]])
        A23 = np.array([[math.cos(q[2]), 0, math.sin(q[2]), 0.082*math.cos(q[2])], 
                        [math.sin(q[2]), 0, -math.cos(q[2]), 0.082*math.sin(q[2])], 
                        [0, 1, 0, (self.z3+self.z4)], 
                        [0, 0, 0, 1]])
        A34 = np.array([[math.cos(q[3]+pi), 0, math.sin(q[3]+pi), 0.083*math.cos(q[3]+pi)], 
                        [math.sin(q[3]+pi), 0, -math.cos(q[3]+pi), 0.083*math.sin(q[3]+pi)], 
                        [0, 1, 0, 0], 
                        [0, 0, 0, 1]])
        A45 = np.array([[math.cos(q[4]), 0, -math.sin(q[4]), 0], [
                        math.sin(q[4]), 0, math.cos(q[4]), 0], 
                        [0, -1, 0, 0.384], 
                        [0, 0, 0, 1]])
        A56 = np.array([[math.cos(q[5]-pi), 0, math.sin(q[5]-pi), 0.088*math.cos(q[5]-pi)], 
                        [math.sin(q[5]-pi), 0, -math.cos(q[5]-pi), 0.088*math.sin(q[5]-pi)], 
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        A67 = np.array([[math.cos(q[6]-pi/4), -math.sin(q[6]-pi/4), 0, 0], 
                        [math.sin(q[6]-pi/4), math.cos(q[6]-pi/4), 0, 0], 
                        [0, 0, 1, 0.21],
                        [0, 0, 0, 1]])
 
        # Compute the transformation matrices between the base and end-effector frames
        T0 = np.matmul(AW0, A01)
        T1 = np.matmul(T0, A12)
        T2 = np.matmul(T1, A23)
        T3 = np.matmul(T2, A34)
        T4 = np.matmul(T3, A45)
        T5 = np.matmul(T4, A56)
        T6 = np.matmul(T5, A67)
        T0e = T6

        # Compute the joint positions of the robot given the input joint angles
        P1 = np.array([0,0,0.141,1])
        P2 = np.matmul(T0, np.array([0,0,0,1]))
        P3 = np.matmul(T1, np.array([0,0,0.195,1]))
        P4 = np.matmul(T2, np.array([0,0,0,1]))
        P5 = np.matmul(T3, np.array([0,0,0.125,1]))
        P6 = np.matmul(T4, np.array([0,0,-0.015,1]))
        P7 = np.matmul(T5, np.array([0,0,0.051,1]))
        P8 = np.matmul(T6, np.array([0,0,0.096,1]))


        # Store the joint positions in a 8x3 numpy array
        jointPositions = np.array([[P1[0], P1[1], P1[2]],
                                [P2[0], P2[1], P2[2]],
                                [P3[0], P3[1], P3[2]],
                                [P4[0], P4[1], P4[2]],
                                [P5[0], P5[1], P5[2]],
                                [P6[0], P6[1], P6[2]],
                                [P7[0], P7[1], P7[2]],
                                [P8[0],P8[1], P8[2]]])

        # Your code ends here
        

        return jointPositions, T0e
       
    # feel free to define additional helper methods to modularize your solution

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
