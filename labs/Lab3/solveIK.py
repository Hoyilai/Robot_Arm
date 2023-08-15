import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.IK_velocity import IK_velocity

from numpy.linalg import inv

class IK:

    # Initialize joint limits
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=500, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """
   

   
        ## STUDENT CODE STARTS HERE
        # Initialize the disp and axis arrays with zeros
        disp = np.zeros(3)
        axis = np.zeros(3)

        # Calculate the disp between the target and current frames
        dx = target[0, 3] - current[0, 3]
        dx = target[0, 3] - current[0, 3]
        dy = target[1, 3] - current[1, 3]
        dz = target[2, 3] - current[2, 3]
        disp = np.array([dx, dy, dz])

        # Calculate the axis of rotation from the current frame to the target frame
        R_current = current[:3, :3]
        R_target = target[:3, :3]

        R_current_inv = inv(R_current)
        R_current_target = R_current_inv @ R_target

        skew_symmetric = (1/2) * (R_current_target - np.transpose(R_current_target))
        a1 = skew_symmetric[2, 1]
        a2 = skew_symmetric[0, 2]
        a3 = skew_symmetric[1, 0]

        axis = np.dot(R_current, np.array([a1, a2, a3]))
        ## END STUDENT CODE

        return disp, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H


        """

        ## STUDENT CODE STARTS HERE

        # Initialize the distance and ang variables with zeros
        distance = 0
        ang = 0

        # Calculate the distance between the origins of G and H
        G_disp = np.array([G[i, 3] for i in range(3)])
        H_disp = np.array([H[i, 3] for i in range(3)])
        distance = np.linalg.norm(H_disp - G_disp)
        
        # Calculate the ang between the orientations of G and H
        R_G = G[:3, :3]
        R_H = H[:3, :3]

        R_G_inv = np.linalg.inv(R_G)
        R_GH = R_G_inv @ R_H

        arccos_input = (np.trace(R_GH) - 1) / 2
        arccos_input = np.clip(arccos_input, -1, 1)

        ang = np.arccos(arccos_input)



        ## END STUDENT CODE

        return distance, ang

    def is_valid_solution(self,q,target):

        ## STUDENT CODE STARTS HERE

        success = True

         # Check if joint angs are within the joint limits
        for idx, angle in enumerate(q):
            if not (IK.lower[idx] <= angle <= IK.upper[idx]):
                success = False
    
        # Calculate the forward kinematics for the given joint angs
        jointPosition, T0e = IK.fk.forward(q)

        # Calculate the distance and ang between the target and current end effector poses
        distance, ang = IK.distance_and_angle(T0e, target)
        
        print("ang: ", ang)
        print("angular_tol: ", self.angular_tol)

         # Check if the distance and ang are within the specified tolerances
        if distance > self.linear_tol:
            success = False
        
        if ang > self.angular_tol:
            success = False

        ## END STUDENT CODE

        return success

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros(7)

        # Calculate the current end effector pose using the given joint angs
        jointPosition, T0e = IK.fk.forward(q)

        # Calculate the disp and axis of rotation between the current and target end effector poses
        disp, axis = IK.displacement_and_axis(target, T0e)

        # Compute the desired joint velocity using the IK_velocity function
        dq = IK_velocity(q, disp, axis)

        ## END STUDENT CODE

        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1):

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []
        current_step = 0
        while True:

            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q,target)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            ## STUDENT CODE STARTS HERE
            
            # Task Prioritization
            dq = np.zeros(7) # TODO: implement me!

            J = calcJacobian(q)
            J_Plus = np.matmul(J.transpose(), np.linalg.inv(np.matmul(J, J.transpose())))
            #I = np.matmul(J, J_Plus)
            I = np.identity(7)
            

            dq = dq_ik + np.matmul((I - np.matmul(J_Plus, J)), dq_center)

            #print(dq)

            # Termination Conditions
            if ((current_step >= self.max_steps) or (np.linalg.norm(dq) < self.min_step_size)): # TODO: check termination conditions
                break # exit the while loop if conditions are met!

            current_step = current_step + 1
            ## END STUDENT CODE

            q = q + dq

        success = self.is_valid_solution(q,target)
        #success = True # for debugging
        return q, success, rollout

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    target = np.array([
        [0,-1,0,0.3],
        [-1,0,0,0],
        [0,0,-1,.5],
        [0,0,0, 1],
    ])

    q, success, rollout = ik.inverse(target, seed)
    

    for i, q in enumerate(rollout):
        joints, pose = ik.fk.forward(q)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    print("Success: ",success)
    print("Solution: ",q)
    print("Iterations:", len(rollout))
