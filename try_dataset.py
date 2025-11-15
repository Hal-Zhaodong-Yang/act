import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from utils import load_data
import matplotlib.pyplot as plt

dataset_dir = '/home/zyang645/RobotLearning/code/act/data/sim_insertion_scripted'

episode_idx = 0
dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
with h5py.File(dataset_path, 'r') as root:
    qpos = root['/observations/qpos'][...]

for i in range(qpos.shape[1]):
    plt.plot(qpos[:, i])

plt.show()


