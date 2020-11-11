import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
import argparse

def get_rank(score_list,index):
    score = score_list[index]
    arr = np.sort(score_list)
    rank = np.where(arr==score)
    return rank[0][-1]

# calculate the mean rank 1000 query trajectories versus different data transformation
def calc_distance_SR_np(file):
    traj_set = torch.load(file,map_location=torch.device('cpu'))
    traj_set = traj_set.numpy()
    traj_set = traj_set.squeeze(1)

    rank = []
    querynum = 1000
    amount = 0
    dbnum = traj_set.shape[0]
    bar = tqdm(range(querynum))
    for i in bar:
        dist = []
        for j in range(querynum, dbnum - amount):
            dist.append(np.sum(np.absolute((traj_set[i] - traj_set[j]))))
        rank.append(get_rank(dist, i))
        bar.set_description(desc="SR on {}".format(file))
    results = sum(rank) / len(rank)
    print(results,"SR on {} Done!".format(file))

# calculate the mean rank of 1000 query trajectories versus different database size.
def calc_distance_SR_np_2(file,amount):
    traj_set = torch.load(file,map_location=torch.device('cpu'))
    traj_set = traj_set.numpy()
    traj_set = traj_set.squeeze(1)

    rank = []
    querynum = 1000
    dbnum = traj_set.shape[0]
    bar = tqdm(range(querynum))
    for i in bar:
        dist = []
        for j in range(querynum, dbnum - amount):
            dist.append(np.sum(np.absolute((traj_set[i] - traj_set[j]))))
        rank.append(get_rank(dist, i))
        bar.set_description(desc="dbsize {}: SR on {}".format(dbnum - amount, file))
    results = sum(rank) / len(rank)
    print(results, "dbsize {}: SR on {} Done!".format(dbnum - amount, file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vector from SR')
    parser.add_argument("--model_path", default=None,
                        help="Path to sr model")
    args = parser.parse_args()
    path = args.model_path
    files = ['downsample_0.0', 'downsample_0.2', 'downsample_0.3', 'downsample_0.4', 'downsample_0.5', 'downsample_0.6',
             'downsample_0.7', 'distort_0.2', 'distort_0.3', 'distort_0.4', 'distort_0.5', 'distort_0.6', 'distort_0.7']

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)

    # similarity results of different data transformation
    for i in range(len(files)):
        files[i] = "vec/" + files[i] + '_' + path.split("_", 1)[1][:-3]
    pool.map(calc_distance_SR_np, files)

    # similarity results of different dbsize
    amount = [20000,40000,60000,80000]
    tasks = []
    for i in range(4):
        tasks.append((files[0], amount[i]))
    pool.starmap(calc_distance_SR_np_2, tasks)
