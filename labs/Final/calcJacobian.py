import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    joint_positions, T0e = FK.forward("test1",q_in)
    
    jac_dis = np.zeros((7,3))
    
    z_axis = FK.get_axis_of_rotation("test1",q_in)
    
    for i in range(7):
        jac_dis[i,:] = T0e[:3,3] - joint_positions[i,:]
        #T0e[:3,3]
        v = np.cross(z_axis[i,:],jac_dis[i,:])
        
        J[0:3,i] = np.transpose(v)
        
        J[3:6,:] = np.transpose(z_axis)

    return J

# if __name__ == '__main__':
#     q = np.array([0, 0, 0, 0, 0, 0, 0])
#     print(np.round(calcJacobian(q),3))
