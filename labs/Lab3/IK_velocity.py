import numpy as np 
from lib.calcJacobian import calcJacobian


def IK_velocity(current_config, desired_linear_velocity, desired_angular_velocity):

    J = calcJacobian(current_config.flatten())

    desired_velocities = np.concatenate((desired_linear_velocity, desired_angular_velocity), axis = 0)

    valid_indices = ~np.isnan(desired_velocities)

    J = J[valid_indices, :]

    joint_velocities = np.linalg.lstsq(J, desired_velocities[valid_indices], rcond = None)[0]

    return joint_velocities.flatten()
