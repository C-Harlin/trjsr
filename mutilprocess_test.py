import multiprocessing as mp
import numpy as np
from numpy import dot
from numpy.linalg import norm
import csv
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import traj_dist.distance as tdist
from utils import lonlat2meters, meters2lonlat
# from PIL import Image
import random
import pickle
import argparse

def tometer(traj):
    for i in range(traj.shape[0]):
        lon,lat = lonlat2meters(traj[i][0],traj[i][1])
        traj[i][0], traj[i][1]=int(lon),int(lat)
    return traj

def get_rank(score_list,index):
    score = score_list[index]
    arr = np.sort(score_list)
    rank = np.where(arr==score)
    return rank[0][-1]

def downsample(seq, rate):
    sample_seq = []
    for item in seq:
        rand = random.randint(0, 9)/10
        if rand >= rate:
            sample_seq.append(item)
    return np.array(sample_seq)


def distort(seq, rate):
    distort_seq = []
    for item in seq: #对每一个轨迹点
        rand = random.randint(0, 9) / 10
        if rand < rate:
            x, y = lonlat2meters(item[0], item[1])
            x = x + 30 * np.random.normal()
            y = y + 30 * np.random.normal()
            lon, lat = meters2lonlat(x, y)
            distort_seq.append(np.array([lon,lat]))
        else:
            distort_seq.append(item)
    return np.array(distort_seq)
'''
数据经过下采样后可能会出现包含轨迹点数为0的轨迹，将这类数据剔除
'''
def eliminate(file_a,file_b):
    f_a = open('data/downsample_a_0.6.data', 'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open('data/downsample_b_0.6.data', 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()
    a_empty = []
    b_empty = []
    for i, traj_A in enumerate(traj_set_A):
        if traj_A.shape[0] == 0:
            a_empty.append(i)
    for i, traj_B in enumerate(traj_set_B):
        if traj_B.shape[0] == 0:
            b_empty.append(i)
    print(a_empty, b_empty)
    index = a_empty + b_empty
    print(index.sort())

    for i in reversed(index):
        del traj_set_A[i]
    for i in reversed(index):
        del traj_set_B[i]
    with open(file_a, 'wb') as f:
        pickle.dump(traj_set_A, f)
    with open(file_b, 'wb') as f:
        pickle.dump(traj_set_B, f)
'''
Test on SR model
'''

def calc_distance_SR(file_a,file_b):
    file_a = file_a[5:-5]
    file_b = file_b[5:-5]
    traj_set_A = torch.load(file_a)
    traj_set_B = torch.load(file_b)
    mae_loss = nn.L1Loss(reduction='sum').cpu()
    cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6).cpu()
    rank = {'MAE':[],'Cosine':[]}
    results = {'MAE':[],'Cosine':[]}
    bar = tqdm(traj_set_A)
    for i, traj_A in enumerate(bar):
        dist = {'MAE':[],'Cosine':[]}
        for traj_B in traj_set_B:
            dist['MAE'].append(mae_loss(traj_A, traj_B).item())
            dist['Cosine'].append(1 - cos_loss(torch.unsqueeze(traj_A,0), torch.unsqueeze(traj_B,0)).item())
        rank['MAE'].append(get_rank(dist['MAE'], i))
        rank['Cosine'].append(get_rank(dist['Cosine'], i))
        bar.set_description(desc="SR on {}{}".format(file_a[:-5], file_a[-3:]))
    results['MAE'] = sum(rank['MAE']) / len(rank['MAE'])
    results['Cosine'] = sum(rank['Cosine']) / len(rank['Cosine'])

    name = file_a[:-5] + file_a[-3:]
    with open('statistic/test_sr_result_{}.csv'.format(name), 'w')as f:
        print("test result of {}".format(name))
        print(results)
        f_csv = csv.DictWriter(f, results.keys())
        f_csv.writeheader()
        f_csv.writerow(results)
    print("SR on {} Done!".format(name))
def calc_distance_SR_2(file_a,file_b):
    file_a = file_a[5:-5]+"_2"
    file_b = file_b[5:-5]+"_2"
    traj_set_A = torch.load(file_a)
    traj_set_B = torch.load(file_b)
    traj_set_A = traj_set_A.cpu().numpy()
    traj_set_B = traj_set_B.cpu().numpy()

    rank = {'MAE': [], 'Cosine': []}
    results = {'MAE': [], 'Cosine': []}
    bar = tqdm(traj_set_A)
    for i, traj_A in enumerate(bar):
        dist = {'MAE': [], 'Cosine': []}
        for traj_B in traj_set_B:
            dist['MAE'].append(np.sum(np.absolute((traj_A - traj_B))))
            dist['Cosine'].append(1 - dot(traj_A[0], traj_B[0]) / (norm(traj_A) * norm(traj_B)))
        rank['MAE'].append(get_rank(dist['MAE'], i))
        rank['Cosine'].append(get_rank(dist['Cosine'], i))
        bar.set_description(desc="SR on {}{}".format(file_a[:-5], file_a[-3:]))
    results['MAE'] = sum(rank['MAE']) / len(rank['MAE'])
    results['Cosine'] = sum(rank['Cosine']) / len(rank['Cosine'])

    name = file_a[:-5] + file_a[-3:]+"_2"
    # with open('statistic/test_sr_result_{}.csv'.format(name), 'w')as f:
    #     print("test result of {}".format(name))
    #     print(results)
    #     f_csv = csv.DictWriter(f, results.keys())
    #     f_csv.writeheader()
    #     f_csv.writerow(results)
    print(results)
    print("SR on {} Done!".format(name))
# def calc_distance_SR_np(file_a,file_b):
#     file_a = file_a[5:-5]
#     file_b = file_b[5:-5]
#     traj_set_A = torch.load(file_a)
#     traj_set_B = torch.load(file_b)
#     traj_set_A = traj_set_A.cpu().numpy()
#     traj_set_B = traj_set_B.cpu().numpy()
#
#     rank = {'MAE':[],'Cosine':[]}
#     results = {'MAE':[],'Cosine':[]}
#     bar = tqdm(traj_set_A)
#     for i, traj_A in enumerate(bar):
#         dist = {'MAE':[],'Cosine':[]}
#         for traj_B in traj_set_B:
#             dist['MAE'].append(np.sum(np.absolute((traj_A - traj_B))))
#             dist['Cosine'].append(1 -dot(traj_A[0],traj_B[0])/(norm(traj_A)*norm(traj_B)))
#         rank['MAE'].append(get_rank(dist['MAE'], i))
#         rank['Cosine'].append(get_rank(dist['Cosine'], i))
#         bar.set_description(desc="SR on {}{}".format(file_a[:-5], file_a[-3:]))
#     results['MAE'] = sum(rank['MAE']) / len(rank['MAE'])
#     results['Cosine'] = sum(rank['Cosine']) / len(rank['Cosine'])
#
#     name = file_a[:-5] + file_a[-3:]
#     # with open('statistic/test_sr_result_{}.csv'.format(name), 'w')as f:
#     #     print("test result of {}".format(name))
#     #     print(results)
#     #     f_csv = csv.DictWriter(f, results.keys())
#     #     f_csv.writeheader()
#     #     f_csv.writerow(results)
#     print(results)
#     print("SR on {} Done!".format(name))
'''
test on baseline
'''
def myedr(r,s,threshold):
    matrix = np.zeros((r.shape[0]+1,s.shape[0]+1))
    for i in range(1,r.shape[0]+1):
        matrix[i][0] = i

    for j in range(1,s.shape[0]+1):
        matrix[0][j] = j

    for i in range(1,r.shape[0]+1):
        for j in range(1,s.shape[0]+1):
            subcost = 0 if np.linalg.norm(r[i-1]-s[j-1])<=threshold else 1
            matrix[i][j] = min(matrix[i-1][j-1]+subcost,matrix[i][j-1]+1,matrix[i-1][j]+1)
    return matrix[r.shape[0]][s.shape[0]]
def calc_distance_benchmark(file):
    file_a = "data/query_"+ file + ".data"
    file_b = "data/db_"+ file + ".data"
    f_a = open(file_a,'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(file_b, 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    data.extend(traj_set_A)
    data.extend(traj_set_B)
    # ----to meter-------
    for i in tqdm(range(len(data))):
        traj = tometer(data[i])
        # data[i] = traj.astype('int32')
        data[i] = traj
    # -------------------
    traj_set_A = data[:1000]
    traj_set_B = data[1000:]
    rank = {'sspd': [], 'dtw': [], 'edr': [], 'lcss': []}
    bar = tqdm(traj_set_A)
    for i, traj_A in enumerate(bar):
        # if i<11:continue
        max_std_a = np.std(traj_A, axis=0).max()
        # min_std_a = np.std(traj_A, axis=0).min()
        # if i==2:break
        dist = {'sspd': [], 'dtw': [], 'edr': [], 'lcss': []}
        for j, traj_B in enumerate(traj_set_B):
            # if j<6048:continue
            max_std_b = np.std(traj_B, axis=0).max()
            # min_std_b = np.std(traj_B, axis=0).min()
            max_std = max(max_std_a,max_std_b)
            # min_std = min(min_std_a,min_std_b)
            # dist['sspd'].append(tdist.sspd(traj_A, traj_B))
            # dist['dtw'].append(tdist.dtw(traj_A, traj_B))
            # dist['lcss'].append(tdist.lcss(traj_A, traj_B, eps=max_std/4))
            # dist['edr'].append(tdist.edr(traj_A, traj_B, eps=max_std/4))
            # myresult = myedr(traj_A, traj_B, max_std/4)
            length = max(traj_A.shape[0], traj_B.shape[0])
            # myedr(traj_A, traj_B, max_std / 4)
            # dist['edr'].append(tdist.edr(traj_A, traj_B, eps=max_std/4)/length)
            dist['edr'].append(tdist.edr(traj_A, traj_B, eps=max_std/4))

        # rank['sspd'].append(get_rank(dist['sspd'], i))
        # rank['dtw'].append(get_rank(dist['dtw'], i))
        rank['edr'].append(get_rank(dist['edr'], i))
        # rank['lcss'].append(get_rank(dist['lcss'], i))
        # if i != 0 and rank['edr'][-1] != 0 :
        #     print(i)

        bar.set_description(desc="benchmark on {}".format(file))

    with open("data/tdist_edr_"+file+".txt", 'w') as filehandle:
        for listitem in rank['edr']:
            filehandle.write('%s\n' % listitem)
    # with open("data/tdist_lcss_"+file+".txt", 'w') as filehandle:
    #     for listitem in rank['lcss']:
    #         filehandle.write('%s\n' % listitem)
    # print("lcss:{} edr:{}  benchmark on {} Done!".format(sum(rank['lcss']) / len(rank['lcss']),sum(rank['edr']) / len(rank['edr']),file))
    print("edr:{}  benchmark on {} Done!".format(sum(rank['edr']) / len(rank['edr']),file))

def calc_distance_benchmark_efficient(file):
    file_a = "data/query_"+ file + ".data"
    file_b = "data/db_"+ file + ".data"
    f_a = open(file_a,'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(file_b, 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    data.extend(traj_set_A)
    data.extend(traj_set_B)
    # ----to meter-------
    for i in tqdm(range(len(data))):
        traj = tometer(data[i])
        data[i] = traj
    # -------------------
    traj_set_A = data[:1000]
    traj_set_B = data[1000:]
    rank = {'sspd': [], 'dtw': [], 'edr': [], 'lcss': []}
    bar = tqdm(traj_set_A)
    T = np.zeros((len(traj_set_B),len(traj_set_B)))
    for i, traj_A in enumerate(bar):
        if i == 0:
            max_std_a = np.std(traj_A, axis=0).max()
            dist = []
            for traj_B in traj_set_B:
                max_std_b = np.std(traj_B, axis=0).max()
                max_std = max(max_std_a,max_std_b)
                # length = max(traj_A.shape[0],traj_B.shape[0])
                length = max(traj_A.shape[0], traj_B.shape[0])
                dist.append(tdist.edr(traj_A, traj_B, eps=max_std/4)/length)
            #--------table---------
            for n in tqdm(range(len(traj_set_B))):
                # table = []
                table = np.zeros((len(traj_set_B)))
                for m in range(n, len(traj_set_B)):
                    # table.append(abs(dist[n]-dist[m]))
                    table[m] = abs(dist[n]-dist[m])
                    # T[n][m-n] = abs(dist[n]-dist[m])
                T[n] = table
            #----------------------
        else:
            max_std_a = np.std(traj_A, axis=0).max()
            max_std_b = np.std(traj_set_B[i], axis=0).max()
            max_std = max(max_std_a, max_std_b)
            length = max(traj_A.shape[0], traj_set_B[i].shape[0])
            l = tdist.edr(traj_A, traj_set_B[i], eps=max_std/4)/length

            dist = []
            for j ,traj_B in enumerate(traj_set_B):
                target = T[i][j-i] if i <= j else T[j][i]
                if target > 2*l:
                    dist.append(float('inf'))
                else:
                    max_std_b = np.std(traj_B, axis=0).max()
                    max_std = max(max_std_a,max_std_b)
                    length = max(traj_A.shape[0], traj_B.shape[0])
                    dist.append(tdist.edr(traj_A, traj_B, eps=max_std/4)/length)
                    if i <= j:
                        T[i][j-i] = max(abs(dist[j] - l), target)
                    else:
                        T[j][i] = max(abs(dist[j]-l), target)
        rank['edr'].append(get_rank(dist, i))
        bar.set_description(desc="benchmark on {}".format(file))

    # with open("data/tdist_edr_"+file+".txt", 'w') as filehandle:
    #     for listitem in rank['edr']:
    #         filehandle.write('%s\n' % listitem)
    print("edr:{}  benchmark on {} Done!".format(sum(rank['edr']) / len(rank['edr']),file))

def calc_distance_benchmark_2(file, amount):
    file_a = "data/query_"+ file + ".data"
    file_b = "data/db_"+ file + ".data"
    f_a = open(file_a,'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(file_b, 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    data.extend(traj_set_A)
    data.extend(traj_set_B)
    # ----to meter-------
    for i in tqdm(range(len(data))):
        traj = tometer(data[i])
        # data[i] = traj.astype('int32')
        data[i] = traj
    # -------------------
    traj_set_A = data[:1000]
    traj_set_B = data[1000:-(100000-amount)]
    rank = {'sspd': [], 'dtw': [], 'edr': [], 'lcss': []}
    bar = tqdm(traj_set_A)
    for i, traj_A in enumerate(bar):
        # if i<11:continue
        max_std_a = np.std(traj_A, axis=0).max()
        # min_std_a = np.std(traj_A, axis=0).min()
        # if i==2:break
        dist = {'sspd': [], 'dtw': [], 'edr': [], 'lcss': []}
        for traj_B in traj_set_B:
            max_std_b = np.std(traj_B, axis=0).max()
            max_std = max(max_std_a,max_std_b)
            length = max(traj_A.shape[0], traj_B.shape[0])
            # dist['edr'].append(tdist.edr(traj_A, traj_B, eps=max_std / 4) / length)
            dist['edr'].append(tdist.edr(traj_A, traj_B, eps=max_std/4))
        rank['edr'].append(get_rank(dist['edr'], i))
        bar.set_description(desc="dbsize:{} edr".format(amount))

    with open("data/tdist_edr_"+str(amount)+".txt", 'w') as filehandle:
        for listitem in rank['edr']:
            filehandle.write('%s\n' % listitem)
    # with open("data/tdist_lcss_"+file+".txt", 'w') as filehandle:
    #     for listitem in rank['lcss']:
    #         filehandle.write('%s\n' % listitem)
    # print("lcss:{} edr:{}  benchmark on {} Done!".format(sum(rank['lcss']) / len(rank['lcss']),sum(rank['edr']) / len(rank['edr']),file))
    print("edr:{}  dbsize: {} Done!".format(sum(rank['edr']) / len(rank['edr']),amount))

def calc_distance_benchmark_2_efficient(file,amount):
    file_a = "data/query_"+ file + ".data"
    file_b = "data/db_"+ file + ".data"
    f_a = open(file_a,'rb')
    traj_set_A = pickle.load(f_a)
    f_b = open(file_b, 'rb')
    traj_set_B = pickle.load(f_b)
    f_a.close()
    f_b.close()

    data = []
    data.extend(traj_set_A)
    data.extend(traj_set_B)
    # ----to meter-------
    for i in tqdm(range(len(data))):
        traj = tometer(data[i])
        # data[i] = traj.astype('int32')
        data[i] = traj
    # -------------------
    traj_set_A = data[:1000]
    traj_set_B = data[1000:-(100000-amount)]
    rank = {'sspd': [], 'dtw': [], 'edr': [], 'lcss': []}
    D = []
    bar = tqdm(traj_set_A)

    for i, traj_A in enumerate(bar):
        if i == 0:
            max_std_a = np.std(traj_A, axis=0).max()
            dist = []
            for traj_B in traj_set_B:
                max_std_b = np.std(traj_B, axis=0).max()
                max_std = max(max_std_a,max_std_b)
                # length = max(traj_A.shape[0],traj_B.shape[0])
                dist.append(tdist.edr(traj_A, traj_B, eps=max_std/4))
            D.append(dist)
        else:
            # max_std_a = np.std(traj_A, axis=0).max()
            # if D[i-1].count(float('inf')) > count:
            #     count = D[i-1].count(float('inf'))
            #     index = i - 1
            index = np.random.randint(0,i) if i == 1 else np.random.randint(1,i)#random pick a previous index
            max_std_b = np.std(traj_set_A[index], axis=0).max()
            max_std = max(max_std_a, max_std_b)
            # length = max(traj_A.shape[0], traj_set_A[index].shape[0])
            l = tdist.edr(traj_A, traj_set_A[index], eps=max_std/4)

            dist = []
            for j ,traj_B in enumerate(traj_set_B):
                if D[index][j] >= 2*l:
                    dist.append(float('inf'))
                else:
                    max_std_b = np.std(traj_B, axis=0).max()
                    max_std = max(max_std_a,max_std_b)
                    # length = max(traj_A.shape[0], traj_B.shape[0])
                    dist.append(tdist.edr(traj_A, traj_B, eps=max_std/4))
            D.append(dist)
        rank['edr'].append(get_rank(D[i], i))
        bar.set_description(desc="dbsize:{} edr".format(amount))

    with open("data/tdist_edr_" + str(amount) + ".txt", 'w') as filehandle:
        for listitem in rank['edr']:
            filehandle.write('%s\n' % listitem)
    # with open("data/tdist_lcss_"+file+".txt", 'w') as filehandle:
    #     for listitem in rank['lcss']:
    #         filehandle.write('%s\n' % listitem)
    # print("lcss:{} edr:{}  benchmark on {} Done!".format(sum(rank['lcss']) / len(rank['lcss']),sum(rank['edr']) / len(rank['edr']),file))
    print("edr:{}  dbsize: {} Done!".format(sum(rank['edr']) / len(rank['edr']), amount))

def calc_distance_SR_np(file):

    traj_set = torch.load(file,map_location=torch.device('cpu'))
    traj_set = traj_set.numpy()
    traj_set = traj_set.squeeze(1)

    rank = {'MAE': [], 'Cosine': []}
    results = {'MAE': [], 'Cosine': []}
    querynum = 1000
    amount = 0
    dbnum = traj_set.shape[0]
    bar = tqdm(range(querynum))
    for i in bar:
        dist = {'MAE': [], 'Cosine': []}
        for j in range(querynum, dbnum - amount):
            dist['MAE'].append(np.sum(np.absolute((traj_set[i] - traj_set[j]))))
            # dist['Cosine'].append(1 - dot(traj_set[i], traj_set[j]) / (norm(traj_set[i]) * norm(traj_set[j])))
        rank['MAE'].append(get_rank(dist['MAE'], i))
        # rank['Cosine'].append(get_rank(dist['Cosine'], i))
        bar.set_description(desc="SR on {}".format(file))
    results['MAE'] = sum(rank['MAE']) / len(rank['MAE'])
    # results['Cosine'] = sum(rank['Cosine']) / len(rank['Cosine'])
    with open("data/trjsr_"+file[4:].split("_")[0]+"_"+file[4:].split("_")[1]+".txt", 'w') as filehandle:
        for listitem in rank['MAE']:
            filehandle.write('%s\n' % listitem)

    # with open('statistic/test_sr_result_{}.csv'.format(name), 'w')as f:
    #     print("test result of {}".format(name))
    #     print(results)
    #     f_csv = csv.DictWriter(f, results.keys())
    #     f_csv.writeheader()
    #     f_csv.writerow(results)
    print(results,"SR on {} Done!".format(file))
def calc_distance_t2vec_np(file):
    f = h5py.File(file, 'r')
    traj_set = f["layer3"].value

    rank = {'MAE': [], 'Cosine': []}
    results = {'MAE': [], 'Cosine': []}
    querynum = 1000
    amount = 0
    dbnum = traj_set.shape[0]
    bar = tqdm(range(querynum))
    for i in bar:
        dist = {'MAE': [], 'Cosine': []}
        for j in range(querynum, dbnum - amount):
            dist['MAE'].append(np.sum(np.absolute((traj_set[i] - traj_set[j]))))
            # dist['Cosine'].append(1 - dot(traj_set[i], traj_set[j]) / (norm(traj_set[i]) * norm(traj_set[j])))
        rank['MAE'].append(get_rank(dist['MAE'], i))
        # rank['Cosine'].append(get_rank(dist['Cosine'], i))
        bar.set_description(desc="t2vec on {}".format(file))
    results['MAE'] = sum(rank['MAE']) / len(rank['MAE'])
    print(results,"t2vec on {} Done!".format(file))
def calc_distance_SR_np_2(file,amount):

    traj_set = torch.load(file,map_location=torch.device('cpu'))
    traj_set = traj_set.numpy()
    traj_set = traj_set.squeeze(1)

    rank = {'MAE': [], 'Cosine': []}
    results = {'MAE': [], 'Cosine': []}
    querynum = 1000

    dbnum = traj_set.shape[0]
    bar = tqdm(range(querynum))
    for i in bar:
        dist = {'MAE': [], 'Cosine': []}
        for j in range(querynum, dbnum - amount):
            dist['MAE'].append(np.sum(np.absolute((traj_set[i] - traj_set[j]))))
            # dist['Cosine'].append(1 - dot(traj_set[i], traj_set[j]) / (norm(traj_set[i]) * norm(traj_set[j])))
        rank['MAE'].append(get_rank(dist['MAE'], i))
        # rank['Cosine'].append(get_rank(dist['Cosine'], i))
        bar.set_description(desc="dbsize {}: SR on {}".format(dbnum - amount, file))
    results['MAE'] = sum(rank['MAE']) / len(rank['MAE'])
    print(results, "dbsize {}: SR on {} Done!".format(dbnum - amount, file))
    with open("data/trjsr_"+str(dbnum - amount), 'w') as filehandle:
        for listitem in rank['MAE']:
            filehandle.write('%s\n' % listitem)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vector from SR')
    parser.add_argument("--model_path", default=None,
                        help="Path to sr model")
    args = parser.parse_args()
    path = args.model_path
    files = ['downsample_0.0', 'downsample_0.2', 'downsample_0.3', 'downsample_0.4', 'downsample_0.5', 'downsample_0.6',
             'downsample_0.7', 'distort_0.2', 'distort_0.3', 'distort_0.4', 'distort_0.5', 'distort_0.6', 'distort_0.7']
    # files = ['downsample_0.2', 'downsample_0.3', 'downsample_0.4', 'downsample_0.5', 'downsample_0.6','downsample_0.7']
    # files = ['distort_0.2', 'distort_0.3', 'distort_0.4', 'distort_0.5', 'distort_0.6','distort_0.7']

    # calc_distance_benchmark('downsample_0.0')

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)

    # pool.map(calc_distance_benchmark, files)
    #
    # amount = [20000, 40000, 60000, 80000]
    # tasks = []
    # for i in range(4):
    #     tasks.append((files[0], amount[i]))
    # pool.starmap(calc_distance_benchmark_2, tasks)

    # calc_distance_benchmark_2_efficient(files[0],20000)


    #--------map usage---------
    for i in range(len(files)):
        files[i] = "vec/" + files[i] + '_' + path.split("_", 1)[1][:-3]
        # files[i] = files[i] + '_' + "origin"

    pool.map(calc_distance_SR_np, files)

    amount = [20000,40000,60000,80000]

    tasks = []
    for i in range(4):
        tasks.append((files[0], amount[i]))
    pool.starmap(calc_distance_SR_np_2, tasks)



    #--------------------------
    # files_b = ['data/distort_b_0.2.data','data/distort_b_0.4.data','data/distort_b_0.6.data',
    #         'data/downsample_b_0.2.data','data/downsample_b_0.4.data','data/downsample_b_0.6.data']
    # files_a = ['data/distort_a_0.2.data', 'data/distort_a_0.4.data', 'data/distort_a_0.6.data',
    #          'data/downsample_a_0.2.data', 'data/downsample_a_0.4.data', 'data/downsample_a_0.6.data']

    # p1 = mp.Process(target=calc_distance_benchmark, args=(files_a[0], files_b[0]))
    # p2 = mp.Process(target=calc_distance_benchmark, args=(files_a[1], files_b[1]))
    # p3 = mp.Process(target=calc_distance_benchmark, args=(files_a[2], files_b[2]))
    # p4 = mp.Process(target=calc_distance_benchmark, args=(files_a[3], files_b[3]))
    # p5 = mp.Process(target=calc_distance_benchmark, args=(files_a[4], files_b[4]))
    # p6 = mp.Process(target=calc_distance_benchmark, args=(files_a[5], files_b[5]))

    # p1 = mp.Process(target=calc_distance_SR_2, args=(files_a[0], files_b[0]))
    # p2 = mp.Process(target=calc_distance_SR_2, args=(files_a[1], files_b[1]))
    # p3 = mp.Process(target=calc_distance_SR_2, args=(files_a[2], files_b[2]))
    # p4 = mp.Process(target=calc_distance_SR_2, args=(files_a[3], files_b[3]))
    # p5 = mp.Process(target=calc_distance_SR_2, args=(files_a[4], files_b[4]))
    # p6 = mp.Process(target=calc_distance_SR_2, args=(files_a[5], files_b[5]))
    #
    # p7 = mp.Process(target=calc_distance_SR_np, args=(files_a[0], files_b[0]))
    # p8 = mp.Process(target=calc_distance_SR_np, args=(files_a[1], files_b[1]))
    # p9 = mp.Process(target=calc_distance_SR_np, args=(files_a[2], files_b[2]))
    # p10 = mp.Process(target=calc_distance_SR_np, args=(files_a[3], files_b[3]))
    # p11 = mp.Process(target=calc_distance_SR_np, args=(files_a[4], files_b[4]))
    # p12 = mp.Process(target=calc_distance_SR_np, args=(files_a[5], files_b[5]))
    #
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    #
    # p7.start()
    # p8.start()
    # p9.start()
    # p10.start()
    # p11.start()
    # p12.start()

    # wait until processes are finished
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    #
    # p7.join()
    # p8.join()
    # p9.join()
    # p10.join()
    # p11.join()
    # p12.join()
    #
    # print("ALL Done!")
