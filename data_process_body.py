import pickle
import numpy as np
import os
import smplx
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from collections import OrderedDict

root_directory = '/data1/guanren/American_poses'

entries = os.listdir(root_directory)

for entry in entries:
    entry_path = os.path.join(root_directory, entry)
    if os.path.exists('/data1/guanren/American_poses/source_body/'+entry+'.npy'):
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
        

    for i in range(transl_concat.shape[0]):
        transl_concat[i, :] = transl_concat[-1, :]
        global_orient_concat[i, :] = global_orient_concat[-1, :]
        body_pose_concat[i, :9, :] = body_pose_concat[-1, :9, :]

    body_pose_concat = body_pose_concat.reshape(-1,63)

    full_body = 22
    body_keypoint = 14
    hand_keypoint = 30
    full_body_hand = 52

    parent_indices = [
        -1, 0, 0, 1, 2, 3, 4, 0, 7, 7, 8, 9, 10, 11
    ]
    # key_point_indices = [0, 1, 2, 4, 5, 7, 8, 9, 16, 17, 18, 19, 20, 21, 34, 35, 36,22, 23, 24, 25, 26, 27, 31,32,33,28,29,30, 49,50,51, 37, 38, 39,40,41,42,46,47,48, 43, 44,45]
    key_point_indices = [0, 1, 2, 4, 5, 7, 8, 9, 16, 17, 18, 19, 20, 21]

    kwargs = dict(gender='neutral',
            num_betas=10,
            use_face_contour=True,
            flat_hand_mean=False,
            use_pca=False,
            batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    smplx_model = smplx.create(
            './body_models', 'smplx', 
            **kwargs).to(device)

    rotation = R.from_euler('x', 180, degrees=True)
    for i in range(body_pose_concat.shape[0]):
        original_rotation = R.from_rotvec(global_orient_concat[i,:])
        rotated_rotation = rotation * original_rotation
        global_orient_concat[i,:] = rotated_rotation.as_rotvec()
    global_positions = np.zeros((body_pose_concat.shape[0], full_body, 3))
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
            global_positions_temp = global_positions_temp[:22,:]
            # global_positions_temp = np.concatenate((global_positions_temp[:22,:],global_positions_temp[25:55,:]), axis=0)
            global_positions[i] = global_positions_temp

    delta_time = 1 / 30
    global_linear_velocity = np.diff(global_positions, axis=0) / delta_time
    global_linear_velocity = np.concatenate([np.zeros((1, full_body, 3)), global_linear_velocity], axis=0)
    global_linear_velocity = global_linear_velocity[:, key_point_indices]

    global_orient_concat = np.expand_dims(global_orient_concat, axis=1)
    body_pose_concat = body_pose_concat.reshape(-1, 21, 3)
    all_body_pose = np.concatenate([global_orient_concat, body_pose_concat], axis=1)
    rotation = R.from_rotvec(all_body_pose.reshape(-1, 3), degrees=False)
    body_pose_quat = rotation.as_quat().reshape(-1, full_body, 4)
    body_pose_quat = body_pose_quat[:, key_point_indices]
    global_orient_concat = np.squeeze(global_orient_concat, axis=1)

    def calculate_global_angular_velocity(joint_orientations, delta_time):

        num_frames, num_joints, _ = joint_orientations.shape
        global_angular_velocity = np.zeros((num_frames - 1, num_joints, 3))

        for frame in range(num_frames - 1):
            for joint in range(num_joints):

                q1 = R.from_quat(joint_orientations[frame, joint])
                q2 = R.from_quat(joint_orientations[frame + 1, joint])
                relative_rotation = q2 * q1.inv()
                rotvec = relative_rotation.as_rotvec()
                angular_velocity = rotvec / delta_time
                global_angular_velocity[frame, joint] = angular_velocity

        return global_angular_velocity

    global_angular_velocity = calculate_global_angular_velocity(body_pose_quat, delta_time)
    global_angular_velocity = np.concatenate([np.zeros((1, body_keypoint, 3)), global_angular_velocity], axis=0)
    def calculate_local_position_from_global(joint_global_pos, parent_indices):

        num_frames, num_joints, _ = joint_global_pos.shape
        joint_local_pos = np.zeros_like(joint_global_pos)

        for j in range(num_joints):
            parent_index = parent_indices[j]
            
            if parent_index == -1:
                joint_local_pos[:, j, :] = transl_concat
            else:
                joint_local_pos[:, j, :] = joint_global_pos[:, j, :] - joint_global_pos[:, parent_index, :]

        return joint_local_pos


    joint_local_pos = calculate_local_position_from_global(global_positions[:,key_point_indices,:], parent_indices)

    new_order = [
        'Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 
        'R_Knee', 'L_Ankle', 'R_Ankle', 
        'Spine3', 'L_Shoulder','R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
    ]

    skeleton_motion = OrderedDict([
        ('rotation', {'arr': body_pose_quat, 'context': {'dtype': 'float32'}}),
        ('root_translation', {'arr': transl_concat, 'context': {'dtype': 'float32'}}),
        ('global_velocity', {'arr': global_linear_velocity, 'context': {'dtype': 'float32'}}),
        ('global_angular_velocity', {'arr': global_angular_velocity, 'context': {'dtype': 'float32'}}),
        ('skeleton_tree', OrderedDict([
            ('node_names', new_order),
            ('parent_indices', {'arr': np.array(parent_indices), 'context': {'dtype': 'int64'}}),
            ('local_translation', {'arr': joint_local_pos[0], 'context': {'dtype': 'float32'}})
        ])),
        ('is_local', True),
        ('fps', 30),
        ('__name__', 'SkeletonMotion')
    ])

    static_joint = np.load('/home/guanren/expressive-humanoid/ASE/ase/poselib/data/npy/S000003_P0000_T00.npy', allow_pickle=True)
    static_joint = OrderedDict(static_joint.item())
    static_joint_list = [0, 1, 2, 3, 4, 5, 6, 7]

    skeleton_motion['rotation']['arr'][:,static_joint_list,:] = static_joint['rotation']['arr'][0,static_joint_list,:]
    skeleton_motion['root_translation']['arr'][:] = static_joint['root_translation']['arr'][0]
    skeleton_motion['skeleton_tree']['local_translation']['arr'][static_joint_list,:] = static_joint['skeleton_tree']['local_translation']['arr'][static_joint_list,:]
    print('/data1/guanren/American_poses/'+entry+'.npy')
    np.save('/data1/guanren/American_poses/source_body/'+entry+'.npy', skeleton_motion)




