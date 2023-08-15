import numpy as np 
from lib.calcJacobian import calcJacobian

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    v_in = 0.05*v_in #too fast
    vel = np.append(v_in,omega_in, axis=0)
    
    J = calcJacobian(q_in)
    uncon = np.argwhere(~np.isnan(vel))

    J = J[uncon[:,0],:] #uncontrainted conditions
    vel = vel[uncon[:,0],:]
    #print(np.round(J,3))
    #print(np.round(vel,3))
    
    J_inv = np.linalg.pinv(J)
    J_aug = np.append(J,vel, axis=1)
    
    #if (np.size(vel) == 0) or (np.linalg.matrix_rank(J) == np.linalg.matrix_rank(J_aug)): #no soln
        #dq = J_inv @ vel
        #print("case 1")
        #print(dq)
        
    #elif np.linalg.matrix_rank(J) < np.linalg.matrix_rank(J_aug): #infinite or unique soln
        #print("case 2")
    dq = np.linalg.lstsq(J, vel,rcond=None)[0]
    #print(dq)
        
    dq = dq.reshape((7,))
    
    return dq

# if __name__ == '__main__':
#     q = np.array([0, 0, 111110, 0, -100000, 10, 10000000000000])
#     (IK_velocity(q,np.array([np.nan,np.nan,np.nan]),np.array([np.nan,np.nan,np.nan])))
