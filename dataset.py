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
            self.hr_image_filenames = hr_image_filenames[0:1280000]
            self.lr_image_filenames = lr_image_filenames[0:1280000]
            # self.hr_image_filenames = hr_image_filenames[0:200]
            # self.lr_image_filenames = lr_image_filenames[0:200]
        else:
            self.hr_image_filenames = hr_image_filenames[0:320000]
            self.lr_image_filenames = lr_image_filenames[0:320000]
            # self.hr_image_filenames = hr_image_filenames[0:200]
            # self.lr_image_filenames = lr_image_filenames[0:200]


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
隔点采样，得到孪生轨迹
'''
def test_data(num):
    f = h5py.File("./data/traj_array.hdf5", 'r')
    # test_index = int(0.85 * len(f.keys()))
    # with h5py.File('./data/downsample_1.hdf5','w') as f_1:
    print("the size of dataset is %d" % len(f.keys()))

    querydb = h5py.File("querydb.hdf5", 'w')
    querynum = 1000
    print("===> create test dataset")

    for i in tqdm(range(num-querynum)):
        index = len(f.keys()) - num + i
        traj = np.array(f.get('%s' % index))
        traj = np.transpose(traj)
        #----------random---------------
        # traj_a = []
        # traj_b = []
        # for j in range(traj.shape[0]):
        #     traj_a.append(traj[j]) if np.random.rand()<0.5 else traj_b.append(traj[j])
        # if i <querynum:
        #     querydb.create_dataset("query/%s" % i, data=np.array(traj_a), dtype='f')
        #     querydb.create_dataset("db/%s" % i, data=np.array(traj_b), dtype='f')
        # else:
        #     querydb.create_dataset("db/%s" % i, data=np.array(traj_b), dtype='f')
        #-------------------------------
        if i <querynum:
            querydb.create_dataset("query/%s" % i, data=traj[::2], dtype='f')
            querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')
        else:
            querydb.create_dataset("db/%s" % i, data=traj[1::2], dtype='f')
        '''
        if i <querynum:
            querydb.create_dataset("query/%s" % i, data=traj, dtype='f')
            querydb.create_dataset("db/%s" % i, data=traj, dtype='f')
        else:
            querydb.create_dataset("db/%s" % i, data=traj, dtype='f')
        '''
    querydb.create_dataset("query/num", data=querynum, dtype='int32')
    querydb.create_dataset("db/num", data=num - querynum, dtype='int32')
    querydb.create_dataset("num", data=num, dtype='int32')
    querydb.close()
    f.close()
    print("finished")

test_data(101000)