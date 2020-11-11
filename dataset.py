from os import listdir
from os.path import join
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage,Resize
import torch
from utils import region
class DatasetFromFolder(Dataset):
    def __init__(self, hr_dataset_dir, lr_dataset_dir):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(hr_dataset_dir, x) for x in listdir(hr_dataset_dir)]
        self.lr_image_filenames = [join(lr_dataset_dir, x) for x in listdir(lr_dataset_dir)]
    def __getitem__(self, index):
        hr_image = ToTensor()(Image.open(self.hr_image_filenames[index]))
        lr_image = ToTensor()(Image.open(self.lr_image_filenames[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)

class MyDataset(Dataset):
    def __init__(self, hr_dataset_dir, lr_dataset_dir,mode):
        super(MyDataset, self).__init__()
        hr_image_filenames = [join(hr_dataset_dir, x) for x in listdir(hr_dataset_dir)]
        lr_image_filenames = [join(lr_dataset_dir, x) for x in listdir(lr_dataset_dir)]
        if mode =='train':
            # the amount of data used for training
            self.hr_image_filenames = hr_image_filenames[0:1280000]
            self.lr_image_filenames = lr_image_filenames[0:1280000]
        else:
            # the amount of data used for validation
            self.hr_image_filenames = hr_image_filenames[0:320000]
            self.lr_image_filenames = lr_image_filenames[0:320000]


    def __getitem__(self, index):
        hr_image = torch.load(self.hr_image_filenames[index])
        lr_image = torch.load(self.lr_image_filenames[index])
        # lr_image = ToTensor()(Image.open(self.lr_image_filenames[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)

def display_transform():
    return Compose([
        ToPILImage(),
        Resize((region['imgsize_y'],region['imgsize_x'])),#尺寸
        # CenterCrop(400),
        ToTensor()
    ])
'''
Create data for evaluation
'''
def test_data(num):
    f = h5py.File("./data/traj_array.hdf5", 'r')

    querydb = h5py.File("querydb.hdf5", 'w')
    querynum = 1000
    print("===> create test dataset")

    for i in tqdm(range(num-querynum)):
        index = len(f.keys()) - num + i
        traj = np.array(f.get('%s' % index))
        traj = np.transpose(traj)

        if i <querynum:
            querydb.create_dataset("query/%s" % i, data=traj[::2], dtype='f')
            querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')
        else:
            querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')

    querydb.create_dataset("query/num", data=querynum, dtype='int32')
    querydb.create_dataset("db/num", data=num - querynum, dtype='int32')
    querydb.create_dataset("num", data=num, dtype='int32')
    querydb.close()
    f.close()
    print("finished")
