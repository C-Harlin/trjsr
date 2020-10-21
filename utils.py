import math
import json
from PIL import Image, ImageDraw
from collections import Counter
import torch
from torch import nn
import random
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
import pickle
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))


def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat
region = {}.fromkeys(['minlat','maxlat','minlon','maxlon','cellsize','numx','numy'])

with open('hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    region['minlon'], region['minlat'] = lonlat2meters(parameters['region']['minlon'], parameters['region']['minlat'])
    region['maxlon'], region['maxlat'] = lonlat2meters(parameters['region']['maxlon'], parameters['region']['maxlat'])
    region['cellsize'] = parameters['region']['cellsize']
    region['cellsize_lr'] = parameters['region']['cellsize_lr']
    region['imgsize_x'] = parameters['region']['imgsize_x']
    region['imgsize_y'] = parameters['region']['imgsize_y']
    region['imgsize_x_lr'] = parameters['region']['imgsize_x_lr']
    region['imgsize_y_lr'] = parameters['region']['imgsize_y_lr']
    # region['numx'] = int(round(region['maxlon'] - region['minlon'], 6) / region['cellsize'])#每行有numx个cell
    region['numy'] = int(round(region['maxlat'] - region['minlat'], 6) / region['cellsize'])#每列有numy个cell
    region['numx_lr'] = int(round(region['maxlon'] - region['minlon'], 6) / region['cellsize_lr'])  # lr每行的cell
    region['numy_lr'] = int(round(region['maxlat'] - region['minlat'], 6) / region['cellsize_lr'])  # lr每列的cell
    # region['pixelrange_x'] = region['cellsize'] * region['imgsize_x'] / (region['maxlon'] - region['minlon'])# cell所占的像素大小
    # region['pixelrange_y'] = region['cellsize'] * region['imgsize_y'] / (region['maxlat'] - region['minlat'])# cell所占的像素大小
    # region['pixelrange_x_lr'] = region['cellsize_lr'] * region['imgsize_x_lr'] / (region['maxlon'] - region['minlon'])
    # region['pixelrange_y_lr'] = region['cellsize_lr'] * region['imgsize_y_lr'] / (region['maxlat'] - region['minlat'])
    region['pixelrange'] = parameters['region']['pixelrange']
    region['pixelrange_lr'] = parameters['region']['pixelrange_lr']
    # assert region['pixelrange_x'] >= 1
    # assert region['pixelrange_y'] >= 1

def coord2cell(x,y):
    xoffset = (x - region['minlon'])/(region['maxlon']-region['minlon'])* region['imgsize_x']/region['pixelrange']
    yoffset = (region['maxlat'] - y)/(region['maxlat']-region['minlat'])* region['imgsize_y']/region['pixelrange']
    xoffset = int(xoffset)
    yoffset = int(yoffset)
    tmp = region['imgsize_x']/region['pixelrange']
    id = yoffset * tmp + xoffset
    return id
#lr
def coord2cell_lr(x, y):
    xoffset = (x - region['minlon']) / (region['maxlon'] - region['minlon']) * region['imgsize_x_lr'] / region[
        'pixelrange_lr']
    yoffset = (region['maxlat'] - y) / (region['maxlat'] - region['minlat']) * region['imgsize_y_lr'] / region[
        'pixelrange_lr']
    xoffset = int(xoffset)
    yoffset = int(yoffset)
    tmp = region['imgsize_x_lr'] / region['pixelrange_lr']
    id = yoffset * tmp + xoffset
    return id
'''
通过cell坐标计算cell像素块范围
'''
def cell2anchor(xoffset, yoffset, pixel):
    left_upper_point_x = xoffset * pixel
    left_upper_point_y = yoffset * pixel
    right_lower_point_x = left_upper_point_x + pixel - 1
    right_lower_point_y = left_upper_point_y + pixel - 1
    # left_upper_point_x = int(round(region['maxlon'] - region['minlon'], 6) / region['cellsize'])

    return (left_upper_point_x, left_upper_point_y), (right_lower_point_x, right_lower_point_y)

# def draw(seq, index, mode):
#     img = Image.new("L", (region['imgsize_x'],region['imgsize_y']))
#     cellset = Counter(seq).keys() # 覆盖的cell
#     occurrence = Counter(seq).values() # 每个cell出现的次数
#     for i, cell in enumerate(cellset):
#         xoffset = cell % (region['imgsize_x']/region['pixelrange'])
#         yoffset = int(cell / region['numy'])
#         left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange'])
#         grayscale = 55 + list(occurrence)[i] * 40 if list(occurrence)[i] < 6 else 255 #每出现一次增加40像素值
#         shape = [left_upper_point, right_lower_point]
#         ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
#     img.save("./image/{}_hr/{}.png".format(mode, index))

# def draw_lr(seq, index, mode):
#     img = Image.new("L", (region['imgsize_x_lr'],region['imgsize_y_lr']))
#     cellset = Counter(seq).keys() # 覆盖的cell
#     occurrence = Counter(seq).values() # 每个cell出现的次数
#     for i, cell in enumerate(cellset):
#         xoffset = cell % region['numx_lr']
#         yoffset = int(cell / region['numy_lr'])
#         left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange_x_lr'], region['pixelrange_y_lr'])
#         grayscale = 100 + list(occurrence)[i] * 50 if list(occurrence)[i] < 4 else 255 #每出现一次增加50像素值
#         shape = [left_upper_point, right_lower_point]
#         ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
#     img.save("./image/{}_lr/{}.png".format(mode,index))

def viz(traj,index):
    map = Image.open("./data/map.png")
    draw = ImageDraw.Draw(map)
    ypixels = map.height
    xpixels = map.width
    for i in range(traj.shape[1]):
        lon, lat = lonlat2meters(traj[0,i],traj[1,i])
        x = int((lon-region['minlon'])/(region['maxlon']-region['minlon'])*xpixels)
        y = int((region['maxlat']-lat)/(region['maxlat']-region['minlat'])*ypixels)
        # draw.point((x,y),fill=(0,0,0))
        draw.ellipse([(x-1, y-1),(x+1, y+1)],fill=(255,0,0))
    map.save("./scp/viz/{}.png".format(index))

def lr_transform():
    return Compose([
        # ToPILImage(),
        Resize((region['imgsize_y_lr'], region['imgsize_x_lr']), interpolation=Image.BICUBIC),
        ToTensor()
    ])

def draw(seq):
    img = Image.new("L", (region['imgsize_x'],region['imgsize_y']))
    cellset = Counter(seq).keys() # 覆盖的cell
    occurrence = Counter(seq).values() # 每个cell出现的次数
    for i, cell in enumerate(cellset):
        xoffset = cell % (region['imgsize_x']/region['pixelrange'])
        yoffset = cell // (region['imgsize_x']/region['pixelrange'])
        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange'])
        # grayscale = 55 + list(occurrence)[i] * 40 if list(occurrence)[i] < 6 else 255 #每出现一次增加40像素值
        grayscale = 105 + list(occurrence)[i] * 50 if list(occurrence)[i] < 4 else 255 #每出现一次增加40像素值
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    # img.save("./scp/hr/{}.png".format(len(seq)))
    return img
def draw_lr(seq):
    img = Image.new("L", (region['imgsize_x_lr'],region['imgsize_y_lr']))
    cellset = Counter(seq).keys() # 覆盖的cell
    occurrence = Counter(seq).values() # 每个cell出现的次数
    for i, cell in enumerate(cellset):
        xoffset = cell % (region['imgsize_x_lr']/region['pixelrange_lr'])
        yoffset = cell // (region['imgsize_x_lr']/region['pixelrange_lr'])
        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, region['pixelrange_lr'])
        # grayscale = 55 + list(occurrence)[i] * 40 if list(occurrence)[i] < 6 else 255 #每出现一次增加40像素值
        grayscale = 105 + list(occurrence)[i] * 50 if list(occurrence)[i] < 4 else 255 #每出现一次增加40像素值
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    # img.save("./scp/lr/{}.png".format(len(seq)))
    return img

def traj2cell(seq):
    cell_seq = []
    for j in range(seq.shape[1]): #对每一个轨迹点
        x, y = lonlat2meters(seq[0][j], seq[1][j]) #将轨迹点坐标换算成米
        # cell_x, cell_y = coord2cell(x, y) #轨迹点所处cell的坐标
        cell_seq.append(coord2cell(x, y)) #轨迹点所处cell的id
    return cell_seq
def traj2cell_lr(seq):
    cell_seq = []
    for j in range(seq.shape[1]): #对每一个轨迹点
        x, y = lonlat2meters(seq[0][j], seq[1][j]) #将轨迹点坐标换算成米
        # cell_x, cell_y = coord2cell(x, y) #轨迹点所处cell的坐标
        cell_seq.append(coord2cell_lr(x, y)) #轨迹点所处cell的id
    return cell_seq

def traj2cell_test_hr(seq):
    cell_seq = []
    for j in range(seq.shape[0]): #对每一个轨迹点
        x, y = lonlat2meters(seq[j][0], seq[j][1]) #将轨迹点坐标换算成米
        # cell_x, cell_y = coord2cell(x, y) #轨迹点所处cell的坐标
        cell_seq.append(coord2cell(x, y)) #轨迹点所处cell的id
    return cell_seq

def traj2cell_test_lr(seq):
    cell_seq = []
    for j in range(seq.shape[0]): #对每一个轨迹点
        x, y = lonlat2meters(seq[j][0], seq[j][1]) #将轨迹点坐标换算成米
        # cell_x, cell_y = coord2cell(x, y) #轨迹点所处cell的坐标
        cell_seq.append(coord2cell_lr(x, y)) #轨迹点所处cell的id
    return cell_seq

def downsample(seq, rate):
    sample_seq = []
    for item in seq:
        rand = random.randint(0, 9)/10
        if rand >= rate:
            sample_seq.append(item)
    return sample_seq

def distort_lr(seq, rate):
    cell_seq = []
    for j in range(seq.shape[1]): #对每一个轨迹点
        x, y = lonlat2meters(seq[0][j], seq[1][j]) #将轨迹点坐标换算成米
        rand = random.randint(0, 9) / 10
        if rand < rate:
            x = x + 100 * np.random.normal()
            y = y + 100 * np.random.normal()
        cell_seq.append(coord2cell_lr(x,y))
    return cell_seq

def create_dataset(traj, index, mode):
    origin = traj2cell(traj)
    hr_img = ToTensor()(draw(origin))
    downsample_rate = [0,0.2,0.4,0.6]
    distort_rate = [0,0.2,0.4,0.6]
    # viz(traj,index)
    num_1 = 0
    for rate_1 in distort_rate:
        noisetrip_1 = distort_lr(traj, rate_1)
        num_2 = 0
        for rate_2 in downsample_rate:
            noisetrip_2 = downsample(noisetrip_1,rate_2)
            lr_img = ToTensor()(draw_lr(noisetrip_2))

            torch.save(lr_img, "image/src_{}/{}.data".format(mode, index * 16 + num_1 * 4 + num_2))
            torch.save(hr_img, "image/trg_{}/{}.data".format(mode, index * 16 + num_1 * 4 + num_2))
            num_2+=1
        num_1+=1


