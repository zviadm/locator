import csv
import math
import sys

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

# TODO zero out unhabitable areas
# TODO augment with conference room model

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

        if l1 > l2:
            toret.append((r1, r2, 10 ** ((l2 - l1)/(10*N))))
        else:
            toret.append((r2, r1, 10 ** ((l1 - l2)/(10*N))))
    return toret



# readings = get_normalized_readings('../newdata/standing_client2.csv')
# readings = get_normalized_readings('../newdata/standing_drew1.csv')
# readings = get_normalized_readings('../newdata/standing_drew2.csv')
# readings = get_normalized_readings('../newdata/standing_breakuproom1.csv')
# readings = get_normalized_readings('../newdata/standing_breakfastbar1.csv')

fname = '../newdata/standing_drew2.csv'
if len(sys.argv) > 1:
    fname = sys.argv[1]
readings = get_normalized_readings(fname)

NUM_BEST = len(readings)

# readings = get_normalized_readings('../newdata/standing_breakfastbar1.csv')
best = sorted(readings.iteritems(), key=itemgetter(1), reverse=True)[:NUM_BEST]

# generate evenly spaced random samples throughout map
# obs reweight
# resample



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

img = mpimg.imread('../map/part4.png')
imgplot = plt.imshow(img)

MAX_PARTICLES = 200
# XMIN = 800
# XMAX = 1500
# XSTEP = 25
# YMIN = 100
# YMAX = 1000
# YSTEP = 25
XMIN = 0
XMAX = 1500
XSTEP = 100
YMIN = 100
YMAX = 1200
YSTEP = 100
LOG_MIN_PROB=-20

MOTION_STDEV = 40
COMBO_ALPHA = 1000

WAVELENGTH = 0.25
N = 2.1
PIXELS_PER_METER = 40.0

samples = [
    [1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)
    ]


# x, y = zip(*samples)
# plt.plot(x, y, 'o', markersize=2)

def get_distance_from_level(level):
    # from wikipedia
    level = -level
    # if level < 60:
    #     n = 2.1
    # elif level < 80:
    #     n = 2.5
    # else:
    #     n = 2.9
    n = 2.1

    C = 20.0 * math.log(4.0 * math.pi / WAVELENGTH, 10)
    r_in_meters = 10 ** ((level - C) / (10.0 * n))

    r_in_meters = max(2.5, r_in_meters)
    dist_in_meters = math.sqrt(r_in_meters ** 2 - 2.5 ** 2)
    return dist_in_meters

def get_distances_from_readings(router_readings):
    return [(r, get_distance_from_level(l)) for r, l in router_readings]

def distance_observation_probability(router_distances, xy):
    ll = 0
    x, y = xy    
    for r, distance in router_distances:
        x1, y1 = ROUTER_POS[r]

        dist = math.sqrt((x - x1)**2/1600.0 + (y - y1)**2 / 1600.0 + 2.5**2)
        ll += log(max(exp(LOG_MIN_PROB), stats.norm(distance, distance/4.0).pdf(dist)))
        # ll += log(max(exp(LOG_MIN_PROB), stats.norm(distance, DISTANCE_STDEV).pdf(dist)))

    #     print distance, dist, log(max(exp(LOG_MIN_PROB), stats.norm(distance, DISTANCE_STDEV).pdf(dist)))
    # print ll
    # print
    return ll
        

def observation_probability(router_ratios, xy):
    ll = 0.0
    x, y = xy
    for r1, r2, ratio in router_ratios:
        x1, y1 = ROUTER_POS[r1]
        x2, y2 = ROUTER_POS[r2]

        dist1 = math.sqrt((x - x1)**2/1600.0 + (y - y1) ** 2/1600.0 + 2.5**2)
        dist2 = math.sqrt((x - x2)**2/1600.0 + (y - y2) ** 2/1600.0 + 2.5**2)

        if (dist1 / dist2) > 1.4:
            ll += LOG_MIN_PROB

        # TODO height correction
        ll += log(max(exp(LOG_MIN_PROB), stats.norm(ratio, ratio/2.0).pdf(dist1 / dist2)))
        # ll += log(max(exp(LOG_MIN_PROB), stats.norm(ratio, RATIO_STDEV).pdf(dist1 / dist2)))

        # print ratio, dist1 / dist2, log(max(exp(LOG_MIN_PROB), stats.norm(ratio, RATIO_STDEV).pdf(dist1/dist2)))
    return ll

print "readings: "
print "\n".join(str(x) for x in sorted(best))
print

print "distances: "
print "\n".join(str(x) for x in sorted(get_distances_from_readings(best)))
print

ratio_model = partial(observation_probability, router_ratios=get_router_distance_ratios(best))

print "ratios: "
print "\n".join(str(x) for x in sorted(get_router_distance_ratios(best)))
print

distance_model = partial(distance_observation_probability, router_distances=get_distances_from_readings(best))


def combo_model(xy):
    # print exp(ratio_model(xy=xy)), 10000*exp(distance_model(xy=xy))
    return log(exp(ratio_model(xy=xy)) + COMBO_ALPHA*exp(distance_model(xy=xy)))


observation_model = combo_model
if len(sys.argv) > 2:
    if sys.argv[2] == 'distance':
        print 'distance model'
        observation_model = distance_model
    if sys.argv[2] == 'ratio':
        print 'ratio model'
        observation_model = ratio_model

# observation_model = partial(observation_probability, router_ratios=[
#     ("AP-4-01", "AP-4-02", 1.0), 
#     ("AP-4-02", "AP-4-03", 1.0), 
#     ("AP-4-03", "AP-4-01", 1.0), 
#     ],)

# observation_model = partial(distance_observation_probability, router_distances=[("AP-4-01", 11),
#     ("AP-4-02", 11),
#     ("AP-4-03", 11),
#     ])


# print "(500, 700): ",  observation_model(xy=(500, 700))
# print "(500, 1100): ", observation_model(xy=(500, 1100))
# print "(500, 300): ",  observation_model(xy=(500, 300))
# print "(700, 700): ",  observation_model(xy=(700, 700))


# obs reweight
def reweight(samples):
    Z = 0.0
    for i in range(len(samples)):
        weight, xy = samples[i]
        nw = weight * 10**observation_model(xy=xy)
        samples[i][0] = nw
        Z += nw

    for i in range(len(samples)):
        samples[i][0] /= Z


# resample
def resample(samples):
    counts = random.multinomial(min(MAX_PARTICLES, len(samples)), zip(*samples)[0])
    return [list(x) for x in zip(counts, zip(*samples)[1]) if x[0] > 0]

def motion(samples):
    toret = []
    for i in range(len(samples)):
        w, (x, y) = samples[i]
        for j in xrange(w):
            toret.append([1, (x + random.normal(0, MOTION_STDEV), y + random.normal(0, MOTION_STDEV))])
    return toret


reweight(samples)
def draw_contour(samples):
    # draw contour of prob map
    weights, locs = zip(*samples)
    tmp = dict(zip(locs, weights))
    Zs = [[tmp[(x, y)] if (x, y) in tmp else 0.0 for x in range(XMIN, XMAX, XSTEP)] for y in range(YMIN, YMAX, YSTEP)]

    # print len(Zs), len(Zs[0]), Zs[0]

    Cs = plt.contour(range(XMIN, XMAX, XSTEP), range(YMIN, YMAX, YSTEP), Zs)
    cbar = plt.colorbar(Cs)

draw_contour(samples)

for i in range(5):
    # print len(samples), samples[:5]
    samples = resample(samples)
    samples = motion(samples)
    reweight(samples)    
    # print len(samples), samples[:5]



xs, ys = zip(*zip(*samples)[1])
plt.plot(xs, ys, 'o', markersize=5)


for name, (x, y) in ROUTER_POS.iteritems():
    plt.text(x, y, name[-2:], color='red')
# xs, ys = zip(*ROUTER_POS.values())
# plt.plot(xs, ys, 'ro', markersize=10)



plt.savefig('test.png')

