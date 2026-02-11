import numpy as np
import torch

from utils import DexArtDataset, load_dexart_data, load_data

"""
check new dataloader for dexart dataset
"""

# dataset_dir = "./data/toilet/hdf5"
# num_episodes = 30
# camera_names = ['front']
# batch_size_train = 8
# train_dataloader, norm_stats = load_dexart_data(dataset_dir, num_episodes, camera_names, batch_size_train)
# for data in train_dataloader:
#     print(data[0].shape)
#     break

# dataset_dir = "./data/sim_transfer_cube_scripted"
# num_episodes = 50
# camera_names = ['top']
# batch_size_train = 8
# train_dataloader, val_dataloader, norm_stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_train)
# print(norm_stats['example_qpos'].shape)

"""
check new dataset
"""

# data_dir="./data/relocate/actions/traj_0/action.npy"

# data = np.load(data_dir)

# print(data.shape)

"""
check flow data pickle file
"""

# import pickle

# with open("./data/toilet/images/cur_target_features/traj_0.pkl", 'rb') as file:
#     flow = pickle.load(file)

# flow = np.array(flow)
# flow = np.squeeze(flow)
# print(flow.shape)

"""
Check dynamo pickle file
"""

# import torch
# file_name = "./data/toilet/images/dynamo_only_cur_features/traj_0.pkl"
# data = torch.load(file_name, map_location="cpu")
# flow_data = data.detach().cpu().numpy()

# print(flow_data.shape)


"""
Check hardware dataset
"""
import pickle
with open("./data/box_processed_with_interpolation/images/only_cur_features/2.pkl", 'rb') as file:
    flow = pickle.load(file)

flow = np.array(flow)
print(flow.shape)
flow = np.squeeze(flow)
print(flow.shape)
flow = np.vstack((np.repeat(flow[0:1, :], repeats=10, axis=0), flow))
print(flow.shape)
print(flow[:20, :])