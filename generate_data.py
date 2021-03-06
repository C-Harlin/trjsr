import numpy as np
import pickle
import h5py
import random
from tqdm import tqdm
from utils import lonlat2meters, meters2lonlat, traj2cell_test_lr, draw_lr
from dataset import test_data
import torch
from model import MyDiscriminator,MyGenerator
from torchvision.transforms import ToTensor
import multiprocessing as mp
import argparse
import pytorch_ssim
from mutilprocess_test import get_rank
import time

parser = argparse.ArgumentParser(description='Generate vector from SR')
parser.add_argument("--model_path", default=None,
    help="Path to sr model")
args = parser.parse_args()

# downsample at rate
def downsample(seq, rate):
    sample_seq = []
    while len(sample_seq) < 10:
        sample_seq = []
        for item in seq:
            rand = random.randint(0, 9)/10
            if rand >= rate:
                sample_seq.append(item)
    return np.array(sample_seq)

# add noise at rate
def distort(seq, rate):
    distort_seq = []
    for item in seq:
        rand = random.randint(0, 9) / 10
        if rand < rate:
            x, y = lonlat2meters(item[0], item[1])
            x = x + 100 * np.random.normal()
            y = y + 100 * np.random.normal()
            lon, lat = meters2lonlat(x, y)
            distort_seq.append(np.array([lon, lat]))
        else:
            distort_seq.append(item)
    return np.array(distort_seq)
# def lonlat2meters(lon, lat):
#     semimajoraxis = 6378137.0
#     east = lon * 0.017453292519943295
#     north = lat * 0.017453292519943295
#     t = math.sin(north)
#     return int(round(semimajoraxis * east)), int(round(3189068.5 * math.log((1 + t) / (1 - t))))

# convert trajectory coordinates into meters
def tometer(traj):
    for i in range(traj.shape[0]):
        lon,lat = lonlat2meters(traj[i][0],traj[i][1])
        traj[i][0], traj[i][1]=int(lon),int(lat)
    return traj

def readfile(filename):
    querydb = h5py.File(filename, 'r')
    query = []
    db = []
    querynum = querydb["query/num"].value
    dbnum = querydb["db/num"].value
    for i in tqdm(range(dbnum)):
        if i < querynum:
            query.append(np.array(querydb.get('query/%s'%i), dtype="float64"))
            db.append(np.array(querydb.get('db/%s'%i), dtype="float64"))
        else:
            db.append(np.array(querydb.get('db/%s' % i), dtype="float64"))
    return query, db


'''
Apply data transformation to evaluation dataset
'''
def transform(query, db):
    downsample_rate = [0.2,0.3,0.4,0.5,0.6,0.7]
    distort_rate = [0.2,0.3,0.4,0.5,0.6,0.7]

    with open('data/query_downsample_0.0.data','wb') as f:
        pickle.dump(query, f)
    with open('data/db_downsample_0.0.data', 'wb') as f:
        pickle.dump(db, f)

    print("===>processing downsample data")
    for rate in downsample_rate:
        data = []
        for traj_a in query:
            data.append(downsample(traj_a,rate))
        with open('data/query_downsample_{}.data'.format(rate), 'wb') as f:
            pickle.dump(data, f)

        data = []
        for traj_b in db:
            data.append(downsample(traj_b, rate))
        with open('data/db_downsample_{}.data'.format(rate), 'wb') as f:
            pickle.dump(data, f)

    print("===>processing distort data")
    for rate in distort_rate:
        data = []
        for traj_a in query:
            data.append(distort(traj_a, rate))
        with open('data/query_distort_{}.data'.format(rate), 'wb') as f:
            pickle.dump(data, f)

        data = []
        for traj_b in db:
            data.append(distort(traj_b, rate))
        with open('data/db_distort_{}.data'.format(rate), 'wb') as f:
            pickle.dump(data, f)

def transform_parallel(query, db, rate):
    print("===> process rate %f" % rate)

    print("===>processing downsample data")
    # downsample trajectories for query
    data = []
    for traj_a in query:
        data.append(downsample(traj_a,rate))
    with open('data/query_downsample_{}.data'.format(rate), 'wb') as f:
        pickle.dump(data, f)

    # downsample trajectories for database
    data = []
    for traj_b in db:
        data.append(downsample(traj_b, rate))
    with open('data/db_downsample_{}.data'.format(rate), 'wb') as f:
        pickle.dump(data, f)

    print("===>processing distort data")
    # distort trajectories for query
    data = []
    for traj_a in query:
        data.append(distort(traj_a, rate))
    with open('data/query_distort_{}.data'.format(rate), 'wb') as f:
        pickle.dump(data, f)

    # distort trajectories for database
    data = []
    for traj_b in db:
        data.append(distort(traj_b, rate))
    with open('data/db_distort_{}.data'.format(rate), 'wb') as f:
        pickle.dump(data, f)


'''
Obtain the .txt file for the evaluation of baseline methods (LCSS, EDR, EDwP).
The codes for those methods are provided by the authors of EDwP
'''
def data_for_java(files_a, files_b, i):
    f_a = open(files_a[i], 'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(files_b[i], 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    with open('{}.txt'.format(files_a[i].split('_',1)[1][:-5]), 'w') as filehandle:
        print("java: processing {}".format(files_a[i].split('_',1)[1][:-5]))
        data.extend(traj_set_A)
        data.extend(traj_set_B)
        #----to meter-------
        for i in range(len(data)):
            traj = tometer(data[i])
            data[i] = traj.astype('int32')
        #-------------------
        for k, item in enumerate(data):
            listitem = item.tolist()
            if k < 1000:
                for j, traj in enumerate(listitem):
                    traj.append(0 + j * 30.0)
            else:
                for j, traj in enumerate(listitem):
                    traj.append(15 + j * 30.0)
            itemstr = str(listitem)
            itemstr = itemstr.replace('], ', '];')
            itemstr = itemstr.replace(', ', ',')
            itemstr = itemstr.replace(']]', '];]')
            itemstr = '0420 24/11/2000 11:30:41 ' + itemstr
            filehandle.write('%s\n' % itemstr)

'''
Obtain the embedding vectors for the evaluation of Trjsr
'''
def data_for_trjsr(file_a, file_b):
    print("trjsr: processing {}".format(file_a.split('_', 1)[1][:-5]))
    # load data in query and database
    f_a = open(file_a, 'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(file_b, 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    data.extend(traj_set_A)
    data.extend(traj_set_B)

    traj_vec = []
    with torch.no_grad():
        for traj in tqdm(data):
            test_traj = traj2cell_test_lr(traj)
            test_img = ToTensor()(draw_lr(test_traj))

            input = torch.unsqueeze(test_img, 0).cuda()
            sr_img = netG(input)
            vec = netD(sr_img)
            traj_vec.append(vec)

    name = "vec/" + file_a.split('_', 1)[1][:-5] + '_' + path.split("_", 1)[1][:-3]
    torch.save(torch.stack(traj_vec, 0), name)

def test_for_ssim(file_a, file_b, amount):
    print("generate image via trjsr: processing {}".format(file_a.split('_', 1)[1][:-5]))
    f_a = open(file_a, 'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(file_b, 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    data.extend(traj_set_A)
    data.extend(traj_set_B)
    traj_img = []
    with torch.no_grad():
        for traj in tqdm(data):
            test_traj = traj2cell_test_lr(traj)
            test_img = ToTensor()(draw_lr(test_traj))
            input = torch.unsqueeze(test_img, 0).cuda()
            sr_img = netG(input)
            traj_img.append(sr_img.to(torch.device("cpu")))
    name = "vec/image/" + file_a.split('_', 1)[1][:-5] + '_' + path.split("_", 1)[1][:-3]
    torch.save(torch.stack(traj_img, 0), name)
    start_time_1 = time.time()
    traj_set = torch.load("vec/image/downsample_0.0_MyG_3_ssim")
    querynum = 1000
    dbnum = traj_set.shape[0]
    print("ssim: processing {} of dbsize {}".format(file_a.split('_', 1)[1][:-5], dbnum-querynum-amount))

    results = {'MSE': [], 'SSIM': []}
    rank = {'MSE':[],'SSIM':[]}
    ssim_loss = pytorch_ssim.SSIM(window_size=11).cuda()
    print("--- %s seconds ---" % (time.time() - start_time_1))
    with torch.no_grad():
        bar = tqdm(range(querynum))
        for i in bar:
            bar.set_description(desc="dbsize {}".format(dbnum - querynum - amount))
            dist = {'MSE': [], 'SSIM': []}
            for j in range(querynum, dbnum - amount):
                traj_set[i].cuda()
                traj_set[j].cuda()
                dist['SSIM'].append(1 - ssim_loss(traj_set[i] , traj_set[j]).item())
            rank['SSIM'].append(get_rank(dist['SSIM'], i))
        results['SSIM'] = sum(rank['SSIM']) / len(rank['SSIM'])
        print("the SSIM result of DBsize {} is {}".format(dbnum-querynum-amount,results['SSIM']))\

def test_for_mse(file_a, file_b, amount):
    traj_set = torch.load("vec/image/downsample_0.0_MyG_3_ssim",map_location=torch.device('cpu'))
    traj_set = traj_set.numpy()
    traj_set = traj_set.squeeze(1)

    querynum = 1000
    dbnum = traj_set.shape[0]
    print("mse: processing {} of dbsize {}".format(file_a.split('_', 1)[1][:-5], dbnum-querynum-amount))

    results = {'MSE': [], 'SSIM': []}
    rank = {'MSE':[],'SSIM':[]}
    with torch.no_grad():
        bar = tqdm(range(querynum))
        for i in bar:
            bar.set_description(desc="dbsize {}".format(dbnum - querynum - amount))
            dist = {'MSE': [], 'SSIM': []}
            for j in range(querynum, dbnum - amount):
                dist['MSE'].append(np.square(traj_set[i]- traj_set[j]).mean())
            rank['MSE'].append(get_rank(dist['MSE'], i))
        results['MSE'] = sum(rank['MSE']) / len(rank['MSE'])
        print("the MSE result of DBsize {} is {}".format(dbnum-querynum-amount,results['MSE']))
'''
Obtain the .h5 file for the evaluation of t2vec
'''
def data_for_t2vec(files_a, files_b, i):
    f_a = open(files_a[i], 'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(files_b[i], 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []

    querynum = 1000
    dbnum = 100000

    with h5py.File('{}.h5'.format(files_a[i].split('_',1)[1][:-5]), 'w') as f:
        print("t2vec: processing {}".format(files_a[i].split('_',1)[1][:-5]))
        data.extend(traj_set_A)
        data.extend(traj_set_B)

        for j, traj in enumerate(data):
            if j<querynum:
                location="query"
                idx = j + 1
            else:
                location = "db"
                idx = j + 1 - 1000
            f.create_dataset('%s/trips/%s' % (location,idx), data=traj)
            f.create_dataset('%s/names/%s' % (location,idx), data=idx)
        f.create_dataset('query/num', data=querynum)
        f.create_dataset('db/num', data=dbnum)

if __name__ == '__main__':
    # set multiprocessing
    num_cores = int(mp.cpu_count())
    print("This computer has " + str(num_cores) + " cores")
    pool = mp.Pool(num_cores)

    test_data(101000) # create dataset for evaluation
    print("==> read queryfb file")
    query, db = readfile("querydb.hdf5")

    print("==> apply downsample and distort at different rates")
    rate = [0.2,0.3,0.4,0.5,0.6,0.7]
    tasks = []
    random.seed(3)
    np.random.seed(3)
    for i in range(len(rate)):
        tasks.append((query, db, rate[i]))
    pool.starmap(transform_parallel, tasks)

    with open('data/query_downsample_0.0.data','wb') as f:
        pickle.dump(query, f)
    with open('data/db_downsample_0.0.data', 'wb') as f:
        pickle.dump(db, f)

    files_a = ['data/query_distort_0.2.data', 'data/query_distort_0.3.data', 'data/query_distort_0.4.data',
               'data/query_distort_0.5.data', 'data/query_distort_0.6.data', 'data/query_distort_0.7.data',
               'data/query_downsample_0.0.data',
               'data/query_downsample_0.2.data', 'data/query_downsample_0.3.data', 'data/query_downsample_0.4.data',
               'data/query_downsample_0.5.data', 'data/query_downsample_0.6.data', 'data/query_downsample_0.7.data']
    files_b = ['data/db_distort_0.2.data', 'data/db_distort_0.3.data', 'data/db_distort_0.4.data',
               'data/db_distort_0.5.data', 'data/db_distort_0.6.data', 'data/db_distort_0.7.data',
               'data/db_downsample_0.0.data',
               'data/db_downsample_0.2.data', 'data/db_downsample_0.3.data', 'data/db_downsample_0.4.data',
               'data/db_downsample_0.5.data', 'data/db_downsample_0.6.data', 'data/db_downsample_0.7.data']

    #-----------load Trjsr model---------
    path = args.model_path
    print("=> loading checkpoint '{}'".format(path))

    netG = MyGenerator(2)
    netD = MyDiscriminator()

    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    best_vec_loss = checkpoint["best_vec_loss"]
    netG.load_state_dict(checkpoint["netG"])
    netD.load_state_dict(checkpoint["netD"])
    print("the model of epoch %d" % start_epoch)

    if torch.cuda.is_available():
        netG.cuda()
        netD.to(torch.device("cuda:0"))
    else:
        netG.cpu()
        netD.cpu()
    netG.eval()
    netD.eval()
    #---------------------------------------
    # generate output of Trjsr
    for i in range(len(files_a)):
        data_for_trjsr(files_a[i], files_b[i])

    index = [x for x in range(0,len(files_a))]
    tasks = []
    for i in range(len(index)):
        tasks.append((files_a, files_b, index[i]))

    # generate test data for java baseline
    pool.starmap(data_for_java, tasks)
    # generate test data for t2vec
    pool.starmap(data_for_t2vec, tasks)