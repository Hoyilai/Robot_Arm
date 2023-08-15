import numpy as np
# from lib.calculateFK import FK

from math import pi
import math


def calcJacobian(q):
    J = np.zeros((6, 7))

    # STUDENT CODE GOES HERE

    # Pre-compute the transformations for each joint
    AW0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.141],
                    [0, 0, 0, 1]])
    A01 = np.array([[math.cos(q[0]), 0, math.sin(q[0]), 0],
                    [math.sin(q[0]), 0, -math.cos(q[0]), 0],
                    [0, 1, 0, 0.192],
                    [0, 0, 0, 1]])
    A12 = np.array([[math.cos(-q[1]), 0, -math.sin(-q[1]), 0],
                    [math.sin(-q[1]), 0, math.cos(-q[1]), 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])
    A23 = np.array([[math.cos(q[2]), 0, math.sin(q[2]), 0.082 * math.cos(q[2])],
                    [math.sin(q[2]), 0, -math.cos(q[2]), 0.082 * math.sin(q[2])],
                    [0, 1, 0, 0.316],
                    [0, 0, 0, 1]])
    A34 = np.array([[math.cos(q[3] + pi), 0, math.sin(q[3] + pi), 0.083 * math.cos(q[3] + pi)],
                    [math.sin(q[3] + pi), 0, -math.cos(q[3] + pi), 0.083 * math.sin(q[3] + pi)],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])
    A45 = np.array([[math.cos(q[4]), 0, -math.sin(q[4]), 0],
                    [math.sin(q[4]), 0, math.cos(q[4]), 0],
                    [0, -1, 0, 0.384],
                    [0, 0, 0, 1]])
    A56 = np.array([[math.cos(q[5] - pi), 0, math.sin(q[5] - pi), 0.088 * math.cos(q[5] - pi)],
                    [math.sin(q[5] - pi), 0, -math.cos(q[5] - pi), 0.088 * math.sin(q[5] - pi)],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])
    A67 = np.array([[math.cos(q[6] - pi / 4), -math.sin(q[6] - pi / 4), 0, 0],
                    [math.sin(q[6] - pi / 4), math.cos(q[6] - pi / 4), 0, 0],
                    [0, 0, 1, 0.21],
                    [0, 0, 0, 1]])

    # Calculate the transformation matrices for each joint
    T0 = np.matmul(AW0, A01)
    T1 = np.matmul(T0, A12)
    T2 = np.matmul(T1, A23)
    T3 = np.matmul(T2, A34)
    T4 = np.matmul(T3, A45)
    T5 = np.matmul(T4, A56)
    T6 = np.matmul(T5, A67)

    # Calculate the position of each joint
    P1 = np.array([0, 0, 0.141, 1])
    P2 = np.matmul(T0, np.array([0, 0, 0, 1]))
    P3 = np.matmul(T1, np.array([0, 0, 0.195, 1]))
    P4 = np.matmul(T2, np.array([0, 0, 0, 1]))
    P5 = np.matmul(T3, np.array([0, 0, 0.125, 1]))
    P6 = np.matmul(T4, np.array([0, 0, -0.015, 1]))
    P7 = np.matmul(T5, np.array([0, 0, 0.051, 1]))
    joints = np.array([[P1[0], P1[1], P1[2]],
                       [P2[0], P2[1], P2[2]],
                       [P3[0], P3[1], P3[2]],
                       [P4[0], P4[1], P4[2]],
                       [P5[0], P5[1], P5[2]],
                       [P6[0], P6[1], P6[2]],
                       [P7[0], P7[1], P7[2]]])

    # Calculate the rotation matrices for each joint
    R01 = T0[0:3, 0:3]
    R02 = T1[0:3, 0:3]
    R03 = T2[0:3, 0:3]
    R04 = T3[0:3, 0:3]
    R05 = T4[0:3, 0:3]
    R06 = T5[0:3, 0:3]
    R0e = T6[0:3, 0:3]

    # Calculate the position of the end effector
    Pe = T6[0:3, 3]

    # Calculate the linear velocity Jacobian
    Jv7 = np.cross(R06[0:3, 2], (Pe - joints[6]))
    Jv6 = np.cross(R05[0:3, 2], (Pe - joints[5]))
    Jv5 = np.cross(R04[0:3, 2], (Pe - joints[4]))
    Jv4 = np.cross(R03[0:3, 2], (Pe - joints[3]))
    Jv3 = np.cross(R02[0:3, 2], (Pe - joints[2]))
    Jv2 = np.cross(-R01[0:3, 2], (Pe - joints[1]))
    Jv1 = np.cross(np.array([0, 0, 1]).transpose(), (Pe - joints[0]))

    # ACalculate the ngular Velocity Jacobian
    Jw1 = np.array([0, 0, 1])
    Jw2 = np.matmul(R01, np.array([0, 0, -1]))
    Jw3 = np.matmul(R02, np.array([0, 0, 1]))
    Jw4 = np.matmul(R03, np.array([0, 0, 1]))
    Jw5 = np.matmul(R04, np.array([0, 0, 1]))
    Jw6 = np.matmul(R05, np.array([0, 0, 1]))
    Jw7 = np.matmul(R06, np.array([0, 0, 1]))

    # Combine the linear and angular velocity Jacobians
    J = np.array([
        [Jv1[0], Jv2[0], Jv3[0], Jv4[0], Jv5[0], Jv6[0], Jv7[0]],
        [Jv1[1], Jv2[1], Jv3[1], Jv4[1], Jv5[1], Jv6[1], Jv7[1]],
        [Jv1[2], Jv2[2], Jv3[2], Jv4[2], Jv5[2], Jv6[2], Jv7[2]],
        [Jw1[0], Jw2[0], Jw3[0], Jw4[0], Jw5[0], Jw6[0], Jw7[0]],
        [Jw1[1], Jw2[1], Jw3[1], Jw4[1], Jw5[1], Jw6[1], Jw7[1]],
        [Jw1[2], Jw2[2], Jw3[2], Jw4[2], Jw5[2], Jw6[2], Jw7[2]],
        ])

    return J


if __name__ == '__main__':
    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
    print(np.round(calcJacobian(q), 3))
