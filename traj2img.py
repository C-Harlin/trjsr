import h5py
import numpy as np
import math
from utils import create_dataset
from tqdm import tqdm

f = h5py.File("./data/traj_array.hdf5", 'r')
amount = 200000 # the amount of data you expect to use in the experiment
val_index = int(0.8 * amount) # 80% for training and 20% for validation

# generate trajectory image
for i in tqdm(range(amount)):
    traj = np.array(f.get('%s'%i))
    if i < val_index:
        create_dataset(traj, i, "train")
    else:
        create_dataset(traj, i-val_index, "val")
f.close()

