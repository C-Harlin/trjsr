import h5py
import numpy as np
import math
from utils import create_dataset
from tqdm import tqdm

f = h5py.File("./data/traj_array.hdf5", 'r')
# amount = len(f.keys())
amount = 200000
print("total amount of the dataset: {}".format(amount))

val_index = int(0.8 * amount)
test_index = amount
print("for another server")
for i in tqdm(range(amount)): #对每一条轨迹
    traj = np.array(f.get('%s'%i))
    if i < val_index:
        create_dataset(traj, i, "train")
    else:
        create_dataset(traj, i-val_index, "val")
    # elif i < test_index:
    #     create_dataset(traj, i-val_index, "val")
    # else:
    #     create_dataset(traj, i-test_index, "test")


    # if i < val_index:
    #     traj = np.array(f.get('%s' % i))
    #     create_dataset(traj, i, "train")

f.close()

