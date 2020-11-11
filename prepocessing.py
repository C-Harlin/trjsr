import csv
import numpy as np
import h5py
import json

with open('hyper_parameters.json') as json_file:
    parameters = json.load(json_file)
    minlon = parameters['region']['minlon']
    minlat = parameters['region']['minlat']
    maxlon = parameters['region']['maxlon']
    maxlat = parameters['region']['maxlat']
    json_file.close()

def inrange(lon_max,lon_min,lat_max,lat_min):
    if lon_max<maxlon and lon_min>minlon and lat_max<maxlat and lat_min>minlat:
        return True
    else:
        return False

'''
read coordinate data from the CSV file and store them into ndarray
'''
with h5py.File("./data/traj_array.hdf5",'w') as f:
    with open('./data/train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)  ##delimiter=' ', quotechar='|'
        i = 0
        for row in reader:
            if row['MISSING_DATA']=='TRUE' or row['MISSING_DATA']=='True':continue
            line = row['POLYLINE']
            corrdinate_list = line[2:-2].split("],[")
            if len(corrdinate_list) < 60 : continue
            seq = np.zeros((2, len(corrdinate_list)))
            for j, item in enumerate(corrdinate_list):
                seq[0, j], seq[1, j] = [float(corrdinate) for corrdinate in item.split(',')]
            #-------keep trajs only in range-------
            lon_max = np.max(seq[0,])
            lon_min = np.min(seq[0,])
            lat_max = np.max(seq[1,])
            lat_min = np.min(seq[1,])
            if inrange(lon_max, lon_min, lat_max, lat_min):
                dset = f.create_dataset('%s' % i, data=seq, dtype='f')
                i += 1
                if i % 10000 == 0: print("complish:{}/?".format(i))
            #--------------------------------------
        csvfile.close()
    f.close()

print("Finished")



