import os

file_directory = "/home/zyang645/RobotLearning/code/act/data/sim_transfer_cube_aloha"

file_names = os.listdir(file_directory)
print(file_names)

for name in file_names:
    previous_name = os.path.join(file_directory, name)
    new_name = os.path.join(file_directory, 'episode_' + name[5:-5] + '.hdf5')
    os.rename(previous_name, new_name)