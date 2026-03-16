import pickle
import numpy as np
import os
import smplx
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
import tqdm 

from collections import OrderedDict

root_directory = '/data1/guanren/CSL-Daily/csl-daily_pose'

entries = os.listdir(root_directory)

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def calculate_rotation_matrix_to_target(source_vector, target_vector):
    """
    Calculate a rotation matrix to align the source_vector to the target_vector.
    
    Args:
        source_vector (numpy.ndarray): Current vector to rotate (3,).
        target_vector (numpy.ndarray): Target vector to align with (3,).
    
    Returns:
        numpy.ndarray: Rotation matrix (3, 3).
    """
    source_vector = normalize(source_vector)
    target_vector = normalize(target_vector)
    cross_prod = np.cross(source_vector, target_vector)
    dot_prod = np.dot(source_vector, target_vector)
    if np.linalg.norm(cross_prod) < 1e-6:  # Already aligned
        return np.eye(3)
    
    skew_symmetric = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    rotation_matrix = (
        np.eye(3) + skew_symmetric +
        np.dot(skew_symmetric, skew_symmetric) * ((1 - dot_prod) / (np.linalg.norm(cross_prod) ** 2))
    )
    return rotation_matrix

def align_09_to_z_axis(global_coords):
    """
    Rotate the hand's global coordinates so that the line connecting
    Joint 0 and Joint 7 is perpendicular to the XY plane (aligned with the Z-axis).
    
    Args:
        global_coords (numpy.ndarray): Shape (n_joints, 3), global coordinates of joints.
    
    Returns:
        numpy.ndarray: Adjusted global coordinates.
    """
    joint_0 = global_coords[0]
    joint_9 = global_coords[9]
    vector_09 = joint_9 - joint_0

    target_vector = np.array([0, 0, 1])  # Z-axis

    rotation_matrix = calculate_rotation_matrix_to_target(vector_09, target_vector)

    rotated_coords = (rotation_matrix @ global_coords.T).T
    return rotated_coords

def vector_angle_with_x_axis_2d(vector):
    """
    Calculate the angle between a 2D vector (projected onto XY plane) and the positive X-axis.

    Args:
        vector (numpy.ndarray): 3D vector, only the X and Y components will be used.

    Returns:
        float: Angle in radians between the vector and the positive X-axis in the XY plane.
    """
    vector_2d = vector[:2]  # Only use X and Y components
    unit_vector = vector_2d / np.linalg.norm(vector_2d)
    x_axis = np.array([1, 0])  # Positive X-axis in 2D
    cos_angle = np.dot(unit_vector, x_axis)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def rotate_based_on_joint_distribution(global_coords):
    """
    Rotate the coordinates to align the projection of the line connecting joint 0 and joint 12
    in the XY plane with the positive X-axis. The direction from joint 0 to joint 12
    should align with the positive X-axis.

    Args:
        global_coords (numpy.ndarray): Shape (n_joints, 3), global coordinates of joints.

    Returns:
        numpy.ndarray: Adjusted global coordinates.
    """
    def calculate_rotation_matrix_2d(angle):
        """Create a 2D rotation matrix for the given angle (radians)."""
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    # Step 1: Compute the vector from joint 0 to joint 12 in the XY plane
    joint_0 = global_coords[0]
    joint_12 = global_coords[11]
    vector_012 = joint_12[:2] - joint_0[:2]  # Only consider the XY components

    # Step 2: Calculate the angle between this vector and the positive X-axis
    angle_with_x = np.arctan2(vector_012[1], vector_012[0])  # Angle in radians

    # Step 3: Calculate the required rotation to align this vector with the positive X-axis
    # Rotate by the negative of this angle to align the vector with the X-axis
    rotation_matrix = calculate_rotation_matrix_2d(-angle_with_x)

    global_coords[:, :2] = (rotation_matrix[:2, :2] @ global_coords[:, :2].T).T

    return global_coords

for entry in entries:
    entry_path = os.path.join(root_directory, entry)
    if os.path.exists('/data1/guanren/CSL-Daily/'+entry+'_hand_full_pose.npy'):
        continue
    all_body_pose = []
    all_left_hand_pose = []
    all_right_hand_pose = []
    all_transl = []
    all_beta = []
    all_global_orient = []
    for filename in sorted(os.listdir(entry_path)):
        if filename.endswith('.pkl'):
            file_path = os.path.join(entry_path, filename)
            
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

            all_body_pose.append(np.expand_dims(data['smplx_body_pose'],axis=0))
            all_left_hand_pose.append(np.expand_dims(data['smplx_lhand_pose'],axis=0))
            all_right_hand_pose.append(np.expand_dims(data['smplx_rhand_pose'],axis=0))
            all_transl.append(np.expand_dims(data['cam_trans'],axis=0))
            all_beta.append(np.expand_dims(data['smplx_shape'],axis=0))
            all_global_orient.append(np.expand_dims(data['smplx_root_pose'],axis=0))

            file.close()

    if len(all_body_pose) > 0:

        body_pose_concat = np.concatenate(all_body_pose, axis=0)
        left_hand_pose_concat = np.concatenate(all_left_hand_pose, axis=0)
        right_hand_pose_concat = np.concatenate(all_right_hand_pose, axis=0)
        transl_concat = np.concatenate(all_transl, axis=0)
        beta_concat = np.concatenate(all_beta, axis=0)
        global_orient_concat = np.concatenate(all_global_orient, axis=0)
        body_pose_concat = body_pose_concat.reshape(-1,21,3)

    else:
        print('No data for '+entry)
        continue

    body_pose_concat = body_pose_concat.reshape(-1, 21, 3)
    for i in range(transl_concat.shape[0]):
        transl_concat[i, :] = transl_concat[-1, :]
        global_orient_concat[i, :] = global_orient_concat[-1, :]
        body_pose_concat[i, :9, :] = body_pose_concat[-1, :9, :]
    body_pose_concat = body_pose_concat.reshape(-1, 63)


    full_body = 22
    body_keypoint = 14
    hand_keypoint = 30
    full_body_hand = 52

    parent_indices = [
        -1, 0, 0, 1, 2, 3, 4, 0, 7, 7, 8, 9, 10, 11
    ]
    # key_point_indices = [0, 1, 2, 4, 5, 7, 8, 9, 16, 17, 18, 19, 20, 21, 34, 35, 36,22, 23, 24, 25, 26, 27, 31,32,33,28,29,30, 49,50,51, 37, 38, 39,40,41,42,46,47,48, 43, 44,45]
    key_point_indices = [0, 1, 2, 4, 5, 7, 8, 9, 16, 17, 18, 19, 20, 21]

    smplx_hand_to_panoptic = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    right_hand_idxs = [21] + list(range(40, 55)) + list(range(71, 76))
    left_hand_idxs = [20] + list(range(25, 40)) + list(range(66, 71))
    kwargs = dict(gender='neutral',
            num_betas=10,
            use_face_contour=True,
            flat_hand_mean=False,
            use_pca=False,
            batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    smplx_model = smplx.create(
            '/home/guanren/expressive-humanoid/body_models', 'smplx', 
            **kwargs).to(device)

    rotation = R.from_euler('x', 180, degrees=True)
    for i in range(body_pose_concat.shape[0]):
        original_rotation = R.from_rotvec(global_orient_concat[i,:])
        rotated_rotation = rotation * original_rotation
        global_orient_concat[i,:] = rotated_rotation.as_rotvec()
    global_positions = np.zeros((body_pose_concat.shape[0], full_body, 3))
    left_global_hands = np.zeros((body_pose_concat.shape[0], 21, 3))
    right_global_hands = np.zeros((body_pose_concat.shape[0], 21, 3))
    for i in range(body_pose_concat.shape[0]):
            betas = beta_concat[i]
            betas = np.expand_dims(betas, axis=0)
            transl = transl_concat[i]
            transl = np.expand_dims(transl, axis=0)
            pose = body_pose_concat[i]
            pose = np.expand_dims(pose, axis=0)
            global_orient = global_orient_concat[i]
            global_orient = np.expand_dims(global_orient, axis=0)
            left_hand_pose = left_hand_pose_concat[i]
            left_hand_pose = np.expand_dims(left_hand_pose, axis=0)
            right_hand_pose = right_hand_pose_concat[i]
            right_hand_pose = np.expand_dims(right_hand_pose, axis=0)

            betas = torch.tensor(betas, dtype=torch.float32).to(device)
            # print(betas.shape)
            transl = torch.tensor(transl, dtype=torch.float32).to(device)
            pose = torch.tensor(pose, dtype=torch.float32).to(device)
            global_orient = torch.tensor(global_orient, dtype=torch.float32).to(device)
            left_hand_pose = torch.tensor(left_hand_pose, dtype=torch.float32).reshape(1,-1).to(device)
            right_hand_pose = torch.tensor(right_hand_pose, dtype=torch.float32).reshape(1,-1).to(device)

            output = smplx_model(betas=betas, body_pose=pose, global_orient=global_orient, transl=transl, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
            global_positions_temp = output.joints.detach().cpu().numpy().squeeze()
            left_hand_joints = global_positions_temp[left_hand_idxs, :][smplx_hand_to_panoptic, :]
            left_joint_pos = left_hand_joints - global_positions_temp[20:20 + 1, :]
            origin = left_joint_pos[0:1, :]
            left_joint_pos -= origin
            left_global_hands[i] = left_joint_pos
            right_hand_joints = global_positions_temp[right_hand_idxs, :][smplx_hand_to_panoptic, :]
            right_joint_pos = right_hand_joints - global_positions_temp[21:21 + 1, :]
            origin = right_joint_pos[0:1, :]
            right_joint_pos -= origin
            right_global_hands[i] = right_joint_pos
            global_positions_temp = global_positions_temp[:22,:]
            # global_positions_temp = np.concatenate((global_positions_temp[:22,:],global_positions_temp[25:55,:]), axis=0)
            global_positions[i] = global_positions_temp

    config_path = get_default_config_path(RobotName.linker, RetargetingType.vector, HandType.left)
    robot_dir = '/home/guanren/expressive-humanoid/body_models/hands'
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    left_retarget_hand = []
    right_retarget_hand = []

    for i in range(global_positions.shape[0]):
        retargeting_type = retargeting.optimizer.retargeting_type
        indices = retargeting.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        temp = align_09_to_z_axis(left_global_hands[i])
        temp = rotate_based_on_joint_distribution(temp)
        ref_value = temp[task_indices, :] - temp[origin_indices, :]
        left_qpos = retargeting.retarget(ref_value)
        left_retarget_hand.append(left_qpos)

    retargeting.verbose()

    left_retarget_hand = np.array(left_retarget_hand)

    config_path = get_default_config_path(RobotName.linker, RetargetingType.vector, HandType.right)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    for i in range(global_positions.shape[0]):
        retargeting_type = retargeting.optimizer.retargeting_type
        indices = retargeting.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        temp = align_09_to_z_axis(right_global_hands[i])
        temp = rotate_based_on_joint_distribution(temp)
        ref_value = temp[task_indices, :] - temp[origin_indices, :]
        right_qpos = retargeting.retarget(ref_value)
        right_retarget_hand.append(right_qpos)

    retargeting.verbose()

    right_retarget_hand = np.array(right_retarget_hand)
    retarget_hand = np.concatenate((left_retarget_hand, right_retarget_hand), axis=1)
    print('/data1/guanren/CSL-Daily/target_linker_hand/'+entry+'_hand_full_pose.npy')
    np.save('/data1/guanren/CSL-Daily/target_linker_hand/'+entry+'_hand_full_pose.npy', retarget_hand)




