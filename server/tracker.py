import csv
import logging
import math
import cStringIO
import sys
import threading
import time

from collections import defaultdict
from operator import itemgetter
from functools import partial

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import log, exp, random, array, mean
from scipy import stats

from tracker_info import update_map_info
# map boundaries to use
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

# variances for different methods
LOG_MIN_PROB=-20
MOTION_STDEV = 25

COMBO_ALPHA = 2
MIN_RATIO_STD = 0.2
MIN_RATIO_STD = 0.5

MAX_PARTICLES = 200

# physical constants for determining path loss (wikipedia)
WAVELENGTH = 0.125
N = 2.1

# map constants
PIXELS_PER_METER = 40.0
MAP_NAME = '../map/part4.png'
IMG = mpimg.imread(MAP_NAME)

ROUTER_POS = {
        # Part 4
        "AP-4-01" : (500, 1105),
        "AP-4-02" : (450, 292),
        "AP-4-03" : (687, 724),
        "AP-4-04" : (1203, 315),
        "AP-4-05" : (1130, 970),

        # Part 3
        "AP-3-01" : (587, 551),
        "AP-3-02" : (921, 285),
        "AP-3-03" : (977, 1131),
        "AP-3-04" : (1297, 709),
        "AP-3-05" : (2004, 300),
        "AP-3-06" : (1996, 1116),

        # Part 2
        "AP-2-01" : (500, 1105),
        "AP-2-02" : (450, 292),
        "AP-2-03" : (687, 724),
        "AP-2-04" : (1203, 315),
        "AP-2-05" : (1130, 970),


        # Part 1
        "AP-1-01" : (589, 514),
        "AP-1-02" : (1029, 284),
        "AP-1-03" : (972, 659),
        "AP-1-04" : (1014, 1092),
        "AP-1-05" : (1741, 296),
        "AP-1-06" : (1611, 1100),
        "AP-1-07" : (2581, 265),
        "AP-1-08" : (2585, 1109),
        }


def get_router_distance_ratios(router_readings):
    NUM_BEST = len(router_readings)

    idx = [(i,j) for i in range(NUM_BEST) for j in range(i+1, NUM_BEST)]
    toret = []
    for i, j in idx:
        r1, l1 = router_readings[i]
        r2, l2 = router_readings[j]

        if l1 > l2:
            toret.append((ROUTER_POS[r1], ROUTER_POS[r2], 10 ** ((l2 - l1)/(10*N))))
        else:
            toret.append((ROUTER_POS[r2], ROUTER_POS[r1], 10 ** ((l1 - l2)/(10*N))))
    return toret

def peval(coeffs, x):
    a, b, c = coeffs
    return a * x ** 2 + b * x + c

def get_distance_from_level(level):
    # n = 2.1
    n = peval([  6.66666667e-05,  -1.06666667e-02,   1.62000000e+00], level)
    # n = peval([  1.16666667e-03,   8.83333333e-02,   3.60000000e+00], level)

    # from wikipedia
    level = -level

    C = 20.0 * math.log(4.0 * math.pi / WAVELENGTH, 10)
    r_in_meters = 10 ** ((level - C) / (10.0 * n))

    r_in_meters = max(2.5, r_in_meters)
    dist_in_meters = math.sqrt(r_in_meters ** 2 - 2.5 ** 2)
    return dist_in_meters

def get_distances_from_readings(router_readings):
    return [(ROUTER_POS[r], get_distance_from_level(l)) for r, l in router_readings]


NORM_Z = log(0.39894)
def loglikelihood(x):
    return NORM_Z - 0.5*x*x

def distance_observation_probability(router_distances, xy):
    ll = 0
    x, y = xy

    for (x1, y1), distance in router_distances:
        dist = math.sqrt((x - x1)**2/1600.0 + (y - y1)**2 / 1600.0)
        ll += max(LOG_MIN_PROB, loglikelihood((distance - dist) / max(MIN_DISTANCE_STD, distance/4.0)))

        # ll += log(max(exp(LOG_MIN_PROB), stats.norm(distance, distance/4.0).pdf(dist)))
    #     print distance, dist, log(max(exp(LOG_MIN_PROB), stats.norm(distance, DISTANCE_STDEV).pdf(dist)))
    # print ll
    # print
    return ll


def ratio_observation_probability(router_ratios, xy):
    ll = 0.0
    x, y = xy
    for (x1, y1), (x2, y2), ratio in router_ratios:

        new_ratio = math.sqrt(((x - x1)**2/1600.0 + (y - y1) ** 2/1600.0 + 2.5**2)/((x - x2)**2/1600.0 + (y - y2) ** 2/1600.0 + 2.5**2))
        if new_ratio > 1.4:
            ll += LOG_MIN_PROB
            continue

        # dist1 = math.sqrt((x - x1)**2/1600.0 + (y - y1) ** 2/1600.0 + 2.5**2)
        # dist2 = math.sqrt((x - x2)**2/1600.0 + (y - y2) ** 2/1600.0 + 2.5**2)
        # if (dist1/dist2) > 1.4:
        #     ll += LOG_MIN_PROB
        #     continue

        # TODO height correction
        ll += max(LOG_MIN_PROB, loglikelihood((new_ratio-ratio)/(max(ratio/2.0, MIN_RATIO_STD))))
        # ll += log(max(exp(LOG_MIN_PROB), stats.norm(ratio, ratio/2.0).pdf(dist1 / dist2)))
    return ll


# obs reweight
def reweight(samples, observation_model):
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

def draw_contour(samples):
    # draw contour of prob map
    weights, locs = zip(*samples)
    tmp = dict(zip(locs, weights))
    Zs = [[tmp[(x, y)] if (x, y) in tmp else 0.0 for x in range(XMIN, XMAX, XSTEP)] for y in range(YMIN, YMAX, YSTEP)]

    # print len(Zs), len(Zs[0]), Zs[0]

    Cs = plt.contour(range(XMIN, XMAX, XSTEP), range(YMIN, YMAX, YSTEP), Zs)
    cbar = plt.colorbar(Cs)

def draw_image(samples):
    plt.cla()
    imgplot = plt.imshow(IMG)

    xs, ys = zip(*zip(*samples)[1])
    plt.plot(xs, ys, 'bo', markersize=5)

    for name, (x, y) in ROUTER_POS.iteritems():
        plt.text(x, y, name[-2:], color='red')

    logging.info("about to write image")
    last_image = cStringIO.StringIO()
    plt.axis([XMIN, XMAX, YMAX, YMIN])
    plt.savefig(last_image, bbox_inches='tight', dpi=75)
    return last_image.getvalue()

device_samples = [
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        ]
device_locks_lock = threading.Lock()
device_locks = {}

def get_mean_and_variance(samples):
    xs, ys = zip(*zip(*samples)[1])
    xs = array(xs)
    ys = array(ys)

    mx = mean(xs)
    my = mean(ys)
    return (mx, my), (math.sqrt(mean((xs-mx).dot(xs-mx))), math.sqrt(mean((ys-my).dot(ys-my))))

def track_location(device_id, timestamp, router_levels):
    global device_samples
    global device_locks

    if not device_id in device_locks:
        with device_locks_lock:
            if not device_id in device_locks:
                device_locks[device_id] = threading.Lock()

    with device_locks[device_id]:
        readings = sorted(router_levels.iteritems(), key=itemgetter(1), reverse=True)
        router_ratios = get_router_distance_ratios(readings)
        router_distances = get_distances_from_readings(readings)

        ratio_model = partial(ratio_observation_probability, router_ratios=router_ratios)
        distance_model = partial(distance_observation_probability, router_distances=router_distances)

        def combo_model(xy):
            # logging.info("%.15f, %.15f" % (exp(ratio_model(xy=xy)), COMBO_ALPHA*exp(distance_model(xy=xy))))
            return ratio_model(xy=xy) + COMBO_ALPHA*distance_model(xy=xy)

        observation_models = [combo_model, distance_model, ratio_model]

        image_data = []
        device_stats = {}
        for i, model in enumerate(observation_models):
            reweight(device_samples[i][device_id], model)
            #draw_contour(device_samples[device_id])
            device_samples[i][device_id] = resample(device_samples[i][device_id])
            device_samples[i][device_id] = motion(device_samples[i][device_id])
            #image_data.append(draw_image(device_samples[i][device_id]))

            mean_xy, var_xy = get_mean_and_variance(device_samples[i][device_id])
            device_stats[device_id + "_" + i] = {
                    "location" : mean_xy,
                    "variance" : var_xy,
                    "color"    : ["blue", "red", "yellow"][i],
                    }

        # update map information
        update_map_info({
            "info" : \
                "readings      : " + " : ".join(("(%s, %6d)" % x) for x in readings) + "\n" + \
                "router dists  : " + " : ".join(("(%s, %6.3f)" % x) for x in router_distances) + "\n\n" + \
                "router_ratios : " + " : ".join(("(%s, %s, %.3f)" % x) for x in router_ratios) + "\n" + \
                "",
            "device_stats" : device_stats,
            "images" : image_data,
            })
