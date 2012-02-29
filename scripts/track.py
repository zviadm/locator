import csv
import math

from collections import defaultdict
from operator import itemgetter
from functools import partial
from numpy import log, exp, random

BSSID_TO_ROUTER = {
        "00:0b:86:74:96:80" : "AP-4-01",
        "00:0b:86:74:96:81" : "AP-4-01",

        "00:0b:86:74:97:60" : "AP-4-02",
        "00:0b:86:74:97:61" : "AP-4-02",

        "00:0b:86:74:9a:80" : "AP-4-03",
        "00:0b:86:74:9a:81" : "AP-4-03",

        "00:0b:86:74:9a:90" : "AP-4-04",
        "00:0b:86:74:9a:91" : "AP-4-04",

        "00:0b:86:74:97:90" : "AP-4-05",
        "00:0b:86:74:97:91" : "AP-4-05",
        }


ROUTER_POS = {
        "AP-4-01" : (500, 1105),
        "AP-4-02" : (450, 292),
        "AP-4-03" : (687, 724),
        "AP-4-04" : (1203, 315),
        "AP-4-05" : (1130, 970),
        }


def normalize_mac(address):
    return ":".join("0"+x if len(x) == 1 else x for x in address.split(":"))    


def get_normalized_readings(fname):
    readings = defaultdict(lambda: -90)
    with open(fname) as f:
        r = csv.reader(f)
        for ssid, m, signal in r:
            mac = normalize_mac(m)
            if mac in BSSID_TO_ROUTER:
                readings[BSSID_TO_ROUTER[mac]] = max(readings[BSSID_TO_ROUTER[mac]], float(signal))
    return readings


def get_router_distance_ratios(router_readings):
    idx = [(i,j) for i in range(NUM_BEST) for j in range(i+1, NUM_BEST)]
    toret = []
    for i, j in idx:
        r1, l1 = router_readings[i]
        r2, l2 = router_readings[j]

        toret.append((r1, r2, 10 ** ((l2 - l1)/(10*2.1))))
    return toret



readings = get_normalized_readings('../newdata/standing_drew2.csv')

NUM_BEST = len(readings)

# readings = get_normalized_readings('../newdata/standing_breakfastbar1.csv')
best = sorted(readings.iteritems(), key=itemgetter(1))[:NUM_BEST]

ratios = get_router_distance_ratios(best)

# generate evenly spaced random samples throughout map
# obs reweight
# resample




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

img = mpimg.imread('../map/part4.png')
imgplot = plt.imshow(img)


XMIN = 800
XMAX = 1500
XSTEP = 25
YMIN = 100
YMAX = 1000
YSTEP = 25
LOG_MIN_PROB=-100
RATIO_STDEV=0.05

MOTION_STDEV = 1

samples = [
    [1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)
    ]

# x, y = zip(*samples)
# plt.plot(x, y, 'o', markersize=2)

def observation_probability(router_ratios, xy):
    ll = 0.0
    for r1, r2, ratio in router_ratios:
        x1, y1 = ROUTER_POS[r1]
        x2, y2 = ROUTER_POS[r2]
        x, y = xy

        dist1 = math.sqrt((x - x1)**2 + (y - y1) ** 2 + 2.5**2)
        dist2 = math.sqrt((x - x2)**2 + (y - y2) ** 2 + 2.5**2)

        # TODO height correction
        ll += log(max(exp(LOG_MIN_PROB), stats.norm(ratio, RATIO_STDEV).pdf(dist1 / dist2)))
    return ll

observation_model = partial(observation_probability, router_ratios=ratios)

# obs reweight
def reweight(samples):
    Z = 0.0
    for i in range(len(samples)):
        weight, xy = samples[i]
        nw = weight * observation_model(xy=xy)
        samples[i][0] = nw
        Z += nw
     
    for i in range(len(samples)):
        samples[i][0] /= Z


# motion


# resample
def resample(samples):
    counts = random.multinomial(min(100, len(samples)), zip(*samples)[0])
    return [list(x) for x in zip(counts, zip(*samples)[1]) if x[0] > 0]

def motion(samples):
    toret = []
    for i in range(len(samples)):
        w, (x, y) = samples[i]
        for j in xrange(w):
            toret.append([1, (x + random.normal(0, MOTION_STDEV), y + random.normal(0, MOTION_STDEV))])
    return toret

for i in range(15):
    print len(samples), samples[:5]
    reweight(samples)
    print len(samples), samples[:5]
    samples = resample(samples)
    print len(samples), samples[:5]
    samples = motion(samples)
    print len(samples), samples[:5]


# draw contour of prob map
# weights, locs = zip(*samples)
# tmp = dict(zip(locs, weights))
# Zs = [[tmp[(x, y)] if (x, y) in tmp else 0.0 for x in range(XMIN, XMAX, XSTEP)] for y in range(YMIN, YMAX, YSTEP)]

# print len(Zs), len(Zs[0]), Zs[0]

# Cs = plt.contour(range(XMIN, XMAX, XSTEP), range(YMIN, YMAX, YSTEP), Zs)
# cbar = plt.colorbar(Cs)



xs, ys = zip(*zip(*samples)[1])
plt.plot(xs, ys, 'o', markersize=5)


plt.savefig('test.png')
