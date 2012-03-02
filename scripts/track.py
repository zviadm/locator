
import csv
import math
import sys

from collections import defaultdict
from operator import itemgetter
from functools import partial
from numpy import log, exp, random, array, mean, std

from server.tracker import get_router_distance_ratios, get_distance_from_level, get_distances_from_readings, loglikelihood, distance_observation_probability, ratio_observation_probability, reweight, resample, motion, draw_contour, get_mean_and_variance, interp_observation_probability, build_model

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
        for record_name, loc_name, ts, ssid, m, signal in r:
            mac = normalize_mac(m)
            if mac in BSSID_TO_ROUTER:
                readings[BSSID_TO_ROUTER[mac]] = max(readings[BSSID_TO_ROUTER[mac]], float(signal))
    return readings

def get_raw_readings(fname):
    readings = defaultdict(lambda: -100)
    with open(fname) as f:
        r = csv.reader(f)
        for record_id, name, ts, ssid, m, signal in r:
            mac = normalize_mac(m)
            readings[mac] = max(readings[mac], float(signal))
    return readings

def get_scan_results(fname):
    readings = defaultdict(lambda: -100)
    results = []
    with open(fname) as f:
        r = csv.reader(f)
        for record_id, name, ts, ssid, m, signal in r:
            results.append({'BSSID': m, 'level': signal, 'SSID': ssid})
            # mac = normalize_mac(m)
            # readings[mac] = max(readings[mac], float(signal))
    return results



fname = '../newdata/standing_drew2.csv'
if len(sys.argv) > 1:
    fname = sys.argv[1]
readings = get_raw_readings(fname)

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
XMIN = 0
XMAX = 1500
XSTEP = 100
YMIN = 100
YMAX = 1200
YSTEP = 100
LOG_MIN_PROB=-20

MOTION_STDEV = 40
COMBO_ALPHA = 2

WAVELENGTH = 0.25
N = 2.1
PIXELS_PER_METER = 40.0

samples = [
    [1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)
    ]


# x, y = zip(*samples)
# plt.plot(x, y, 'o', markersize=2)




# ratio_model = partial(ratio_observation_probability, router_ratios=get_router_distance_ratios(best))

# best = sorted(readings.iteritems(), key=itemgetter(1), reverse=True)[:NUM_BEST]

print "readings: "
print "\n".join(str(x) for x in sorted(best))
print

# print "distances: "
# print "\n".join(str(x) for x in sorted(get_distances_from_readings(best)))
# print

# print "ratios: "
# print "\n".join(str(x) for x in sorted(get_router_distance_ratios(best)))
# print

# distance_model = partial(distance_observation_probability, router_distances=get_distances_from_readings(best))


# def combo_model(xy):
#     # print exp(ratio_model(xy=xy)), 10000*exp(distance_model(xy=xy))
#     # print ratio_model(xy=xy), distance_model(xy=xy)
#     return ratio_model(xy=xy) + COMBO_ALPHA*distance_model(xy=xy)


    # ('../newdata/bromancechamber1.csv', 'bromancechamber', (1160,  155)), # bromance
    # ('../newdata/breakuproom1.csv', 'breakuproom', (1280,  155)), # breakup
    # ('../newdata/arrears1.csv', 'arrears', (1400,  155)), # arrears
    # ('../newdata/molly1.csv', 'molly', (1100,  350)), # molly
    # ('../newdata/client1.csv', 'client', (1400, 450)), # client
    # ('../newdata/drew1.csv', 'drew', (1400, 620)), # drew
    # ('../newdata/breakfastbar1.csv', 'breakfastbar', (1100, 570)), # breakfastbar

TRAINING_DATA = [
    ('../locdata/zviad1.csv', 'zviad1', (1150, 320)), #z1
    ('../locdata/zviad2.csv', 'zviad2', (1150, 620)), #z2
    ('../locdata/zviad3.csv', 'zviad3', (1150, 980)), #z3
    ('../locdata/zviad4.csv', 'zviad4', (680,  950)), #z4
    ('../locdata/zviad5.csv', 'zviad5', (680,  680)), #z5
    ('../locdata/zviad6.csv', 'zviad6', (680,  400)), #z6
    ('../locdata/zviad7.csv', 'zviad7', (1480, 240)), #z7    
    ]

# model_data: location: {normed_device_id: (mean, variance)}
model_data = build_model(TRAINING_DATA)

observation_model = partial(interp_observation_probability, router_readings=get_scan_results(fname), model=model_data)

# observation_model = combo_model
# if len(sys.argv) > 2:
#     if sys.argv[2] == 'distance':
#         print 'distance model'
#         observation_model = distance_model
#     if sys.argv[2] == 'ratio':
#         print 'ratio model'
#         observation_model = ratio_model
#     if sys.argv[2] == 'interp':
#         print 'interp model'
#         observation_model = interp_model




# observation_model(xy=(1000, 300))
# sys.exit()

# observation_model = partial(ratio_observation_probability, router_ratios=[
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




reweight(samples, observation_model)
draw_contour(samples)

for i in range(2):
    # print len(samples), samples[:5]
    samples = resample(samples)
    samples = motion(samples)
    reweight(samples, observation_model)    
    # print len(samples), samples[:5]



xs, ys = zip(*zip(*samples)[1])
plt.plot(xs, ys, 'o', markersize=5)


for name, (x, y) in ROUTER_POS.iteritems():
    plt.text(x, y, name[-2:], color='red')
# xs, ys = zip(*ROUTER_POS.values())
# plt.plot(xs, ys, 'ro', markersize=10)


(mx, my), (vx, vy) = get_mean_and_variance(samples)
print "m, v: ", (mx, my), (vx, vy)

plt.plot(mx, my, 'ko', markersize=(vx*vy/30000))

plt.savefig('test.png')

