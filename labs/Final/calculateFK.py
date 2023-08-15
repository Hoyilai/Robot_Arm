import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

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

        M = np.empty((4,4,8))
        
        a = [0 , 0 , 0.0825 , -0.0825 , 0 , 0.088 , 0]
        d = [0.333 , 0 , 0.3160 , 0 , 0.384 , 0  , 0.21]
        alph = [3*np.pi/2 , np.pi/2 , np.pi/2 , 3*np.pi/2 , np.pi/2 , np.pi/2 ,0]
        
        for i in range(6):
            M[:,:,i] = [[np.cos(q[i]) , -np.sin(q[i])*np.cos(alph[i]) , np.sin(q[i])*np.sin(alph[i]) , a[i]*np.cos(q[i])],
                        [np.sin(q[i]) , np.cos(q[i])*np.cos(alph[i]) , -np.cos(q[i])*np.sin(alph[i]) , a[i]*np.sin(q[i])],
                        [0 , np.sin(alph[i]) , np.cos(alph[i]) , d[i] ],
                        [0 , 0 , 0 , 1]]
            
        M[:,:,6] = [[np.cos(q[6]-np.pi/4) , -np.sin(q[6]-np.pi/4)*np.cos(alph[6]) , np.sin(q[6]-np.pi/4)*np.sin(alph[6]) , a[6]*np.cos(q[6]-np.pi/4)],
        [np.sin(q[6]-np.pi/4) , np.cos(q[6]-np.pi/4)*np.cos(alph[6]) , -np.cos(q[6]-np.pi/4)*np.sin(alph[6]) , a[6]*np.sin(q[6]-np.pi/4)],
        [0 , np.sin(alph[6]) , np.cos(alph[6]) , d[6] ],
        [0 , 0 , 0 , 1]]
        

        Jp4 = np.zeros((4,1))
        jointPositions = np.empty((8,3))
        
        base = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.141],
            [0, 0, 0, 1]])
        
        Jp4 = base @ np.array([0,0,0,1])#red
        jointPositions[0,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ np.array([0,0,0,1])#green
        jointPositions[1,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ np.array([-0.0825,-0.121,0,1]) #blue
        jointPositions[2,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ np.array([0.0825,0,0,1])#cyan
        jointPositions[3,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ np.array([0,-0.259,0,1])#pink
        jointPositions[4,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ M[:,:,5] @ np.array([-0.088,-0.015,0,1])#yellow
        jointPositions[5,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ M[:,:,5] @ M[:,:,6] @ np.array([0,0,-0.159,1])#yellow
        jointPositions[6,:] = np.transpose(Jp4[0:3])
        
        Jp4 = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ M[:,:,5] @ M[:,:,6] @ np.array([0,0,0,1])#yellow
        jointPositions[7,:] = np.transpose(Jp4[0:3])
        
        T0e = M[:,:,0]@M[:,:,1]@M[:,:,2]@M[:,:,3]@M[:,:,4]@M[:,:,5]@M[:,:,6]
        
        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1


    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        M = np.empty((4,4,8))

        base = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.141],
            [0, 0, 0, 1]])
        
        a = [0 , 0 , 0.0825 , -0.0825 , 0 , 0.088 , 0]
        d = [0.333 , 0 , 0.3160 , 0 , 0.384 , 0  , 0.21]
        alph = [3*np.pi/2 , np.pi/2 , np.pi/2 , 3*np.pi/2 , np.pi/2 , np.pi/2 ,0]
        
        for i in range(6):
            M[:,:,i] = [[np.cos(q[i]) , -np.sin(q[i])*np.cos(alph[i]) , np.sin(q[i])*np.sin(alph[i]) , a[i]*np.cos(q[i])],
                        [np.sin(q[i]) , np.cos(q[i])*np.cos(alph[i]) , -np.cos(q[i])*np.sin(alph[i]) , a[i]*np.sin(q[i])],
                        [0 , np.sin(alph[i]) , np.cos(alph[i]) , d[i] ],
                        [0 , 0 , 0 , 1]]
            
        M[:,:,6] = [[np.cos(q[6]-np.pi/4) , -np.sin(q[6]-np.pi/4)*np.cos(alph[6]) , np.sin(q[6]-np.pi/4)*np.sin(alph[6]) , a[6]*np.cos(q[6]-np.pi/4)],
        [np.sin(q[6]-np.pi/4) , np.cos(q[6]-np.pi/4)*np.cos(alph[6]) , -np.cos(q[6]-np.pi/4)*np.sin(alph[6]) , a[6]*np.sin(q[6]-np.pi/4)],
        [0 , np.sin(alph[6]) , np.cos(alph[6]) , d[6] ],
        [0 , 0 , 0 , 1]]
        
        z_axis = np.zeros((7,4))
        z_axis[0,:] = base @ np.array([0,0,1,0])
        z_axis[1,:] = M[:,:,0] @ np.array([0,0,1,0])
        z_axis[2,:] = M[:,:,0] @ M[:,:,1] @ np.array([0,0,1,0])
        z_axis[3,:] = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ np.array([0,0,1,0])
        z_axis[4,:] = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ np.array([0,0,1,0])
        z_axis[5,:] = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ np.array([0,0,1,0])
        z_axis[6,:] = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ M[:,:,5] @ np.array([0,0,1,0])
        #z_axis[,:] = M[:,:,0] @ M[:,:,1] @ M[:,:,2] @ M[:,:,3] @ M[:,:,4] @ M[:,:,5] @ M[:,:,6] @ np.array([0,0,1,0])
        z_axis = np.round(z_axis[:,0:3],3)
        
        return z_axis

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

# if __name__ == "__main__":

#     fk = FK()

#     # matches figure in the handout
#     q = np.array([0,0,0,0,0,0,0])

#     joint_positions, T0e = fk.forward(q)
#     ate = fk.get_axis_of_rotation(q)

#     print("Joint Positions:\n",ate)
#     print("End Effector Pose:\n",T0e)