import sys
import numpy as np
from copy import deepcopy
from math import pi

from lib.solveIK import IK
from lib.calculateFK import FK

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from core.utils import transform
from time import sleep

static_tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6']
dynamic_tags = ['tag7', 'tag8', 'tag9', 'tag10', 'tag11', 'tag12']

lower_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upper_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

import sys
import numpy as np
from copy import deepcopy
from math import pi

from lib.solveIK import IK

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

static_tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6']
dynamic_tags = ['tag7', 'tag8', 'tag9', 'tag10', 'tag11', 'tag12']

lower_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upper_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

def transBlock(dyn_blocks):
    """
    Calculates the translation of the end effector needs to take to get the 
    dynamic block
    :param: dyn_block - 4x4xn array containing array pose name and its respective pose
    :return: trans_block - 4x1 array containing the translational component of the transformation matrix
    """
    (name, pose) = dyn_blocks[0] # define the pose of a given block
    blk_pose_ee = H_ee_camera @ pose  # static block in ee frame

    angle = (np.sign(np.trace(blk_pose_ee[:3,:3])))*np.arccos((np.trace(blk_pose_ee[:3,:3])-1)/2)

    blk_R = np.array([[np.cos(angle), -np.sin(angle), 0], # rotate about z-axis
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])


    blk_pose_glo = current_pose @ blk_pose_ee # static block in global frame
    corr_blk_R = blk_R
    curr_R = current_pose[:3, :3] # the rotation of the ee in global frame

    trans_curr = current_pose[:, 3] # the translation of the ee in global frame
    trans_block = blk_pose_glo[:, 3] # the translation of the block in global frame

    blk_R_zaxis = blk_R[2, :] # isolate 3rd row

    return trans_block


def closestGrabPose(new_block_pose, hover_pose):
    """
    Since there are 4 possible orientations for the end effector to grab the blocks at, this 
    function calculates the most optimal one to reduce time 
    :param new_block_pos: 
    :param numBlk: int - the number of blocks already stack in that mode(position)
    :param team: string - team "red" vs "blue"
    :return: q - joint angles to place the block in a given pose
    """
    z_rot_90CCW = np.array([[0,-1,0, 0],[1,0,0, 0],[0,0,1, 0], [0, 0, 0, 1]]) # rotation matix for 90 degree counter clockwise about z-axis

    q_choices = np.zeros((4,7)) # array storing joint angles for all possible orientations
    q_change = np.zeros(4,) # array storing the norm of the difference between current joint angles and each choice

    new_block_pose_H = new_block_pose

    choices = np.zeros((4,4,4))
    min_dis = 100

    for i in range(4):
        q, dis, ang, _,_ = ik.inverse(new_block_pose_H, hover_pose)
        
        min_dis = dis
        tol = 0.001
        if dis > tol: # distance tolerance
            q = np.array([100, 100, 100, 100, 100, 100, 100])

        q_choices[i,:] =  q - hover_pose # joint angles choices
        q_change[i] = np.linalg.norm(q_choices[i,:]) # norm of the choice (smaller = better)

        choices[:,:,i] = new_block_pose #the end effector pose of that choice

        new_block_pose = new_block_pose @ z_rot_90CCW

    ind = np.argsort(q_change) #sort each choice by its norm

    # sometimes a solution may have the smallest change in ee orientation but its error be non-negligible
    success = (min_dis < tol) 
    # the choice with ind[0] is the best ee orientation with the 
    return choices[:,:,ind[0]], success 

def placeBlock(mode,numBlk,team): 
    """
    Calulates the joint angles to place the block in a certain pose on the goal area
    :param mode: boolean - switches between two possible locations on the goal area since we
    are going for the two stack stategy
    :param numBlk: int - the number of blocks already stack in that mode(position)
    :param team: string - team "red" vs "blue"
    :return: q - joint angles to place the block in a given pose
    """

    offset = 0.05 # distance along x between two possible positions(modes) in goal area 

    if (team == 'blue') and (mode == 0):
        goal_area = np.array([
            [1, 0, 0, 0.570+offset],
            [0, -1, 0, -0.169],
            [0, 0, -1, 0.25],
            [0, 0, 0, 1],
        ])
    if (team == 'blue') and (mode == 1):
        goal_area = np.array([
            [1, 0, 0, 0.570-offset],
            [0, -1, 0, -0.169],
            [0, 0, -1, 0.25],
            [0, 0, 0, 1],
        ])
    elif (team == 'red') and (mode == 0):
        goal_area = np.array([
            [1, 0, 0, 0.570+offset],
            [0, -1, 0, 0.169],
            [0, 0, -1, 0.25],
            [0, 0, 0, 1],
        ])
    elif (team == 'red') and (mode == 1):
        goal_area = np.array([
            [1, 0, 0, 0.570-offset],
            [0, -1, 0, 0.169],
            [0, 0, -1, 0.25],
            [0, 0, 0, 1],
        ])

    # increases the height of the end effect by adding numBlk times the length of the block's side
    goal_area[2,3] = numBlk*0.07 + goal_area[2,3]

    q, _, _, _, _ = ik.inverse(goal_area,start_position)

    return q


if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array(
        [-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + pi / 2, 0.75344866])
    arm.safe_move_to_position(start_position)  # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n")  # go!

    # STUDENT CODE HERE

    # Define the goal area
    goal_area = np.zeros((4, 4))

    goal_area[0:3, 0:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    goal_area[3, 3] = 1

    hover_target = np.zeros((4, 4))


    if team == 'blue':
        goal_area = np.array([
            [1, 0, 0, 0.600],
            [0, -1, 0, -0.169],
            [0, 0, -1, 0.26],
            [0, 0, 0, 1],
        ])
        hover_target = np.array([  # hover over static blocks intially
            [1, 0, 0, 0.515],
            [0, -1, 0, 0.169],
            [0, 0, -1, 0.5],
            [0, 0, 0, 1],
        ])
    else:
        goal_area = np.array([
            [1, 0, 0, 0.600],
            [0, -1, 0, 0.169],
            [0, 0, -1, 0.26],
            [0, 0, 0, 1],
        ])
        hover_target = np.array([  # hover over static blocks intially
            [1, 0, 0, 0.515],
            [0, -1, 0, -0.169],
            [0, 0, -1, 0.5],
            [0, 0, 0, 1],
        ])

    ik = IK()

    hover_pose, _, _, _, _ = ik.inverse(hover_target, start_position)
    arm.safe_move_to_position(hover_pose)

    # get the transform from camera to panda_end_effector

    # H_ee_camera = cameraFilter(10) #insert the number of samples you want for the camera median filter
    H_ee_camera = detector.get_H_ee_camera()

    current_pose = hover_target  # ee in global frame
    # Detect some blocks...
    i = 0
    mode = 0
    numBlk = 0

    all_poses = np.zeros((4,4,4))
    y_vals = np.zeros((4,))
    mat = np.zeros((4,4))

    for (name, pose) in detector.get_detections():
        blk_pose_ee = H_ee_camera @ pose  # static block in end effector frame
        blk_pose_glo = current_pose @ blk_pose_ee # static block in global frame 
        curr_R = current_pose[:3, :3] # extract the rotational part of the current ee frame

        blk_R_ee = blk_pose_glo[:3, :3] # extract the rotational part of the block pose in the flo frame
        trans_curr = current_pose[:, 3] # extract the translation part of the ee in the glo frame
        trans_block = blk_pose_glo[:, 3]  # extract the translation part of the block in glo frame

        y_vals[i] = trans_block[1] 
        mat[:3,:3] = blk_R_ee
        mat[:,3] = trans_block
        all_poses[:,:,i] = mat

        print(np.round(all_poses[:,:,i],3))
        i += 1

    sort_ind = np.argsort(abs(y_vals))

    all_poses_ee = all_poses[:,:,sort_ind]

    for i in range(4): #Iterate through all 4 static blocks 
        blk_ee = all_poses_ee[:,:,i]
        blk_R_ee = blk_ee[:3,:3]

        # since the block's z-axis is not always norm to the surface, the block's frame must be redefined
        # find the columns of the pose that does not corresponse to a one as this is the first axis of the new pose
        first_axis = blk_R_ee[:,~np.isclose(abs(blk_R_ee[2,:]),1)][:,0] 

        rot = np.zeros((4,4))

        z_axis = np.array([0,0,-1]) # the end effector's z-axis points down

        # take the cross product between first axis and z-axis to get the 2nd axis in the block's frame
        scnd_axis = np.cross(z_axis,first_axis) 

        # define a new frame 
        rot[:3,:3] = np.concatenate((first_axis.reshape(3,1),scnd_axis.reshape(3,1),z_axis.reshape(3,1)),axis = 1)
 
        success = False
        seed = hover_pose

        rot[:,3] = blk_ee[:,3] # translational component 

        rot, success = closestGrabPose(rot, seed)

        print("Block frame in EE frame",np.round(rot,3))

        rot[2, 3] += 0.01

        arm.exec_gripper_cmd(8e-2, 0) 

        q, _, _, _, _ = ik.inverse(rot, hover_pose) 

        arm.safe_move_to_position(q) # align EE

        arm.exec_gripper_cmd(2e-2, 75) # grab the block
        
        mode = (mode+1) % 2 # switch between modes (0,1,0,...)

        q = placeBlock(mode,np.floor(numBlk),team)
        numBlk += 0.5 # number of blocks increases as follows: 0,0,1,1,2...

        arm.safe_move_to_position(q)

        goal_area[2, 3] += i * 0.050

        arm.exec_gripper_cmd(10e-2, 0)
        q[6] -= 2*pi/5 # rotate the EE slightly
        arm.safe_move_to_position(q)

        q, dis, ang, _, _ = ik.inverse(hover_target, start_position)
        arm.safe_move_to_position(q) # go back to hover position

        i += 1

    # tacking dynamic blocks now!!!
    hover_target_dynamic = np.zeros((4, 4))
    sleept = 0

    # sleep times were calibrated in robot lab and were different for each side
    if team == 'blue':
        sleept = 1
        trans = np.array([0, -0.75, 0.35])
        rotation = np.array([0, pi, -2.7])
        hover_target_dynamic = transform(trans, rotation)
    else:
        sleept = 2.5
        trans = np.array([0, 0.75, 0.35])
        rotation = np.array([0, pi, 0])
        hover_target_dynamic = transform(trans, rotation)


    fk = FK()
    hover_dyn, _, _, _, _ = ik.inverse(hover_target_dynamic, start_position)
    # copy_hover_dyn = hover_dyn
    if team =="blue":
        hover_dyn[0] -= 0.05
        hover_dyn[3] -= 0.05

    # get the transform from camera to panda_end_effector

    current_pose = hover_target_dynamic  # ee in global frame

    # detect some blocks...
    while True:

        arm.safe_move_to_position(hover_dyn)
        arm.exec_gripper_cmd(10e-2, 0)

        dyn_blocks = []
        detector.detections = []

        # while the camera does not detect any blocks, this loop should be repeated
        while (len(dyn_blocks) == 0):
            dyn_blocks = detector.get_detections()

            # when a block is detected, we should check if they satisfy constraints
            if (len(dyn_blocks) != 0):

                block_x = transBlock(dyn_blocks)[0] # check first block from the list
                print(block_x)
                if team == 'blue':
                    # if the block is on blue, we need to make sure that the block is incoming into the camera's vision
                    if (block_x < 0.06) or (block_x < -10): 
                        dyn_blocks = []
                        detector.detections = [] # if constraints ar enot satisfied, we must clear the detector's reading
                else:
                    # if the block is on red, we need to make sure that the block is incoming into the camera's vision
                    if (block_x > -0.07) or (block_x > 10):
                        print("aaaa")
                        dyn_blocks = []
                        detector.detections = [] # if constraints ar enot satisfied, we must clear the detector's reading

        trans_block = transBlock(dyn_blocks)

        rot = np.zeros((4,4))

        ee_frame = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

        # calculate the radius of the circle the dyn block takes
        radius = np.sqrt(trans_block[0] ** 2 + (0.990 - abs(trans_block[1])) ** 2)

        rot = np.zeros((4,4))
        grasp_trans = np.zeros((4, 1))
        grasp_rot = np.zeros((3, 3))

        # calculate the translation of the end-effector based on the target block's trajectory's radius.
        # it is different for each team; note: we are not considering the orientation of the block
        if team == 'blue':
            grasp_trans = np.array([0, (radius - 0.990), 0.23, 1])
            grasp_rot = np.array([0, pi, -2.7]) # same orientation as hover_dyn
        else:
            grasp_trans = np.array([0, (0.990 - radius), 0.23, 1])
            grasp_rot = np.array([0, pi, 0]) # same orientation as hover_dyn


        # transform to get the robot's desired pose
        grasp_mat = transform(grasp_trans, grasp_rot)
        grasp, _, _, _, _ = ik.inverse(grasp_mat, hover_dyn)

        # before moving to the pose, delay the action to match the block's timing.
        # note, sleept was calibrated for each side during robot lab
        sleep(sleept)
        arm.safe_move_to_position(grasp)

        arm.exec_gripper_cmd(0.01, 75)

        sleep(1)

        # gripper's state is checked to see if the end-effector successfully grasped a block or not
        gripper = arm.get_gripper_state()
        gripper_pos = gripper['position']
        gripper_width = abs(gripper_pos[0] - gripper_pos[1])


        print(gripper_pos)

        # if the end-effector is almost fully closed, then it should skip to the next iteration, without moving to the goal area
        if  gripper_width < 0.02:
            continue

        # if the end effector successfully grasped block, place block in goal area
        i += 1
        mode = (mode+1) % 2
        q = placeBlock(mode, np.floor(numBlk) ,team)
        numBlk += 0.5

        arm.safe_move_to_position(q)

        arm.exec_gripper_cmd(10e-2, 0)
        q[6] -= pi/3
        arm.safe_move_to_position(q)