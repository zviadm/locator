import csv
import pickle

from numpy import exp
from collections import defaultdict

LOCS = [
    ('../newdata/bromancechamber1.csv', (160,  25)), # bromance
    ('../newdata/breakuproom1.csv', (195,  25)), # breakup
    ('../newdata/arrears1.csv', (230,  25)), # arrears
    ('../newdata/molly1.csv', (160,  80)), # molly
    ('../newdata/client1.csv', (220, 110)), # client
    ('../newdata/drew1.csv', (220, 155)), # drew
    ('../newdata/breakfastbar1.csv', (150, 150)), # breakfastbar
    ]

data = defaultdict(list)
for fname, (x, y) in LOCS:
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for ssid, device_id, strength in reader:
            if True or 'Dropbox' in ssid:
                data[device_id].append((x, y, float(strength)))


sampling_grid = [(x, y) for x in range(140, 250, 5) for y in range(10, 170, 5)]

sample_points = [
    (x, y, R) for x, y in sampling_grid for R in range(-30, -81, -1)
    ]

cache = {}

alpha_x = 0.002
alpha_y = 0.002
alpha_R = 0.01



device_id = data.keys()[0]
print device_id
cache[device_id] = {}
for x in range(140, 250, 5):
    print x
    for y in range(10, 170, 5):
        for R in range(-30, -81, -1):
            for xd, yd, Rd in data[device_id]:
                Kx = exp(-(alpha_x * (x - xd) ** 2 + alpha_y * (y - yd) ** 2 + alpha_R * (R - Rd) ** 2))
                # if Kx == 0:
                #     print "0-ed out"
                cache[device_id][(x, y, R)] = 1.0/len(data[device_id])*Kx


with open('cached_kde_map', 'wb') as f:
    pickle.dump(cache, f)
