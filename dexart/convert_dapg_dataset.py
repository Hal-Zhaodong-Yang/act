import numpy as np
import torch
import os
import h5py
from PIL import Image
import pickle

data_root = "./data/relocate"
trajectory_list = os.listdir(data_root + "/actions")
trajectory_num = len(trajectory_list)

# dataset_dim = {"./data/relocate":{'flow': 256, 'proprioception': 30, 'action': 30}}

for i in range(trajectory_num):
    obs_data = np.load(data_root + f"/observations/traj_{i}/observation.npy")
    action_data = np.load(data_root + f"/actions/traj_{i}/action.npy")
    # For door task, need to change cur_target_features to only_cur_features
    with open(data_root + f"/images/cur_target_features/traj_{i}.pkl", 'rb') as file:
        flow_data = pickle.load(file)
    flow_data = np.squeeze(np.array(flow_data))

    # For door task, need to change to dynamo_only_cur_features, for other tasks, it is dynamo_cur_target_features
    dynamo_file_name = data_root + f"/images/dynamo_cur_target_features_128/traj_{i}.pkl"
    dynamo_data = torch.load(dynamo_file_name, map_location="cpu")
    dynamo_flow = dynamo_data.detach().cpu().numpy()

    new_dataset_path = data_root + "/hdf5"
    new_file_path = new_dataset_path + f"/episode_{i}"

    max_timesteps = obs_data.shape[0]

    # img_sequence = []
    # for img_idx in range(max_timesteps):
    #     img_path = data_root + f"/images/traj_{i}/frame_{img_idx:0{4}d}.png"
    #     img = Image.open(img_path)
    #     img_sequence.append(np.asarray(img))
    # img_sequence = np.array(img_sequence)

    with h5py.File(new_file_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # root.attrs['sim'] = True
        obs = root.create_group('observations')
        # image = obs.create_group('images')
        # front_img = image.create_dataset('front', (max_timesteps, 480, 640, 3), dtype='uint8',
        #                         chunks=(1, 480, 640, 3), )
        proprioception = obs.create_dataset('proprioception', (max_timesteps, obs_data.shape[1]))
        flow = obs.create_dataset('flow', (max_timesteps, flow_data.shape[1]))
        dynamo = obs.create_dataset('dynamo', (max_timesteps, 256))
        action = root.create_dataset('actions', (max_timesteps, action_data.shape[1]))

        root['observations/proprioception'][...] = obs_data[0:max_timesteps, :]
        root['observations/flow'][...] = flow_data[0:max_timesteps, :]
        root['observations/dynamo'][...] = dynamo_flow[0:max_timesteps, :]
        # action dataset contains initial observation as the first timestep, which is used for Koopman training
        root['actions'][...] = action_data[1:max_timesteps+1, :]
        # root['observations/images/front'][...] = img_sequence

