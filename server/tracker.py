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
from numpy import log, exp, random, array, mean, dot, std
from scipy import stats

from tracker_info import update_map_info

# map boundaries to use
XMIN = 250
XMAX = 5500
XSTEP = 250
YMIN = 50
YMAX = 750
YSTEP = 50

# variances for different methods
LOG_MIN_PROB = -20
MOTION_STDEV = 25

COMBO_ALPHA = 2
MIN_RATIO_STD = 0.2
MIN_DISTANCE_STD = 0.2
MAX_PARTICLES = 400

# physical constants for determining path loss (wikipedia)
WAVELENGTH = 0.125
# N_COEFFS = [-0.07192023, -2.40415772]
N_COEFFS = [-0.07363796, -2.52218124]

# map constants
ROUTER_HEIGHT = 2.1
PIXELS_PER_METER = 22.0
PIXELS_PER_METER_SQ = PIXELS_PER_METER**2
ROUTER_POS = {
        # Part 4
        "AP-4-01" : (4980, 600),
        "AP-4-02" : (4980, 200),
        "AP-4-03" : (5041, 409),
        "AP-4-04" : (5325, 159),
        "AP-4-05" : (5265, 537),

        # "AP-4-01" : (500, 1105),
        # "AP-4-02" : (450, 292),
        # "AP-4-03" : (687, 724),
        # "AP-4-04" : (1203, 315),
        # "AP-4-05" : (1130, 970),

        # Part 3
        # "AP-3-01" : (587, 551),
        # "AP-3-02" : (921, 285),
        # "AP-3-03" : (977, 1131),
        # "AP-3-04" : (1297, 709),
        # "AP-3-05" : (2004, 300),
        # "AP-3-06" : (1996, 1116),

        # Part 2
        # "AP-2-01" : (500, 1105),
        # "AP-2-02" : (450, 292),
        # "AP-2-03" : (687, 724),
        # "AP-2-04" : (1203, 315),
        # "AP-2-05" : (1130, 970),


        # Part 1

        # "AP-1-06" : (1611, 1100),
        # "AP-1-07" : (2581, 265),
        # "AP-1-08" : (2585, 1109),
        }

def lin_eval(coeffs, x):
    a, b = coeffs
    return a * x + b

def compute_N(level):
    return max(2, lin_eval(N_COEFFS, level))


def get_router_distance_ratios(router_readings):
    NUM_BEST = len(router_readings)

    idx = [(i,j) for i in range(NUM_BEST) for j in range(i+1, NUM_BEST)]
    toret = []
    for i, j in idx:
        r1, l1 = router_readings[i]
        r2, l2 = router_readings[j]

        avgN = 0.5*(compute_N(l1) + compute_N(l2))
        if l1 > l2:
            toret.append((ROUTER_POS[r1], ROUTER_POS[r2], 10 ** ((l2 - l1)/(10*avgN))))
        else:
            toret.append((ROUTER_POS[r2], ROUTER_POS[r1], 10 ** ((l1 - l2)/(10*avgN))))
    return toret


def get_distance_from_level(level):
    n = compute_N(level)

    # from wikipedia
    level = -level

    C = 20.0 * math.log(4.0 * math.pi / WAVELENGTH, 10)
    r_in_meters = 10 ** ((level - C) / (10.0 * n))

    r_in_meters = max(ROUTER_HEIGHT, r_in_meters)
    dist_in_meters = math.sqrt(r_in_meters ** 2 - ROUTER_HEIGHT ** 2)
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
        dist = math.sqrt((x - x1)**2/PIXELS_PER_METER_SQ + (y - y1)**2 / PIXELS_PER_METER_SQ)
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

        new_ratio = math.sqrt(((x - x1)**2/PIXELS_PER_METER_SQ + (y - y1) ** 2/PIXELS_PER_METER_SQ + ROUTER_HEIGHT**2)/((x - x2)**2/PIXELS_PER_METER_SQ + (y - y2) ** 2/PIXELS_PER_METER_SQ + ROUTER_HEIGHT**2))
        if new_ratio > 1.4:
            ll += LOG_MIN_PROB
            continue

        # dist1 = math.sqrt((x - x1)**2/PIXELS_PER_METER_SQ + (y - y1) ** 2/PIXELS_PER_METER_SQ + ROUTER_HEIGHT**2)
        # dist2 = math.sqrt((x - x2)**2/PIXELS_PER_METER_SQ + (y - y2) ** 2/PIXELS_PER_METER_SQ + ROUTER_HEIGHT**2)
        # if (dist1/dist2) > 1.4:
        #     ll += LOG_MIN_PROB
        #     continue

        # TODO height correction
        ll += max(LOG_MIN_PROB, loglikelihood((new_ratio-ratio)/(max(ratio/2.0, MIN_RATIO_STD))))
        # ll += log(max(exp(LOG_MIN_PROB), stats.norm(ratio, ratio/2.0).pdf(dist1 / dist2)))
    return ll

def normalize_mac(address):
    return ":".join("0"+x if len(x) == 1 else x for x in address.split(":"))

TRAINING_DATA = [
    ('../locdata/zviad1.csv', 'zviad1', (5295, 196)), #z1
    ('../locdata/zviad2.csv', 'zviad2', (5295, 353)), #z2
    ('../locdata/zviad3.csv', 'zviad3', (5285, 554)), #z3
    ('../locdata/zviad4.csv', 'zviad4', (5029, 544)), #z4
    ('../locdata/zviad5.csv', 'zviad5', (5029, 396)), #z5
    ('../locdata/zviad6.csv', 'zviad6', (5032, 237)), #z6
    ('../locdata/zviad7.csv', 'zviad7', (5478, 152)), #z7
    ]
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

# model_data: location: {normed_device_id: (mean, variance)}
model_data = {}
for training_fname, label, location in TRAINING_DATA:
    data = defaultdict(list)
    with open(training_fname, 'rb') as f:
      reader = csv.reader(f)
      for recording_id, loc_name, ts, ssid, device_id, strength in reader:
        if True or 'Dropbox' in ssid:
            m = normalize_mac(device_id)
            if m in BSSID_TO_ROUTER:
                data[BSSID_TO_ROUTER[m]].append(float(strength))
    # NSAMPLES=10
    model_data[location] = dict((key, (mean(val), max(0.2, std(val)))) for key, val in data.iteritems())


def distancesq(xy1, xy2):
    x1, y1 = xy1
    x2, y2 = xy2
    return (x1-x2)**2 + (y1-y2)**2

def interp_observation_probability(router_readings, xy):
    (dist1sq, loc1, l1), (dist2sq, loc2, l2) = sorted((distancesq(xy, location), device_dict, location) for location, device_dict in model_data.iteritems())[:2]

    # print "closest locs: ", dist1sq, l1
    # print "closest locs: ", dist2sq, l2

    dist1 = math.sqrt(dist1sq)
    dist2 = math.sqrt(dist2sq)

    alpha = dist2/(dist1+dist2)

    ll = 0.0
    # couldnt = 0
    for device, signal in router_readings.iteritems():
        if device in loc1 and device in loc2:
            p1 = loglikelihood((signal - loc1[device][0])/(loc1[device][1]))
            p2 = loglikelihood((signal - loc2[device][0])/(loc2[device][1]))
            ll += max(LOG_MIN_PROB, alpha*p1 + (1-alpha)*p2)
            continue
            # mu = loc1[device][0] * alpha + loc2[device][0] * (1-alpha)
            # sigma = loc1[device][1] * alpha + loc2[device][1] * (1-alpha)
            # ll += max(LOG_MIN_PROB, loglikelihood((signal - mu) / sigma))
            # continue
        else:
            # couldnt += 1
            # ll += LOG_MIN_PROB
            continue
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

# def draw_image(samples):
#     plt.cla()
#     imgplot = plt.imshow(IMG)

#     xs, ys = zip(*zip(*samples)[1])
#     plt.plot(xs, ys, 'bo', markersize=5)

#     for name, (x, y) in ROUTER_POS.iteritems():
#         plt.text(x, y, name[-2:], color='red')

#     logging.info("about to write image")
#     last_image = cStringIO.StringIO()
#     plt.axis([XMIN, XMAX, YMAX, YMIN])
#     plt.savefig(last_image, bbox_inches='tight', dpi=75)
#     return last_image.getvalue()

device_samples = [
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        defaultdict(lambda: [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]),
        ]
device_scan_results = {}
device_locks_lock = threading.Lock()
device_locks = {}

def get_mean_and_variance(samples):
    xs, ys = zip(*zip(*samples)[1])
    xs = array(xs)
    ys = array(ys)

    mx = mean(xs)
    my = mean(ys)
    return (mx, my), (math.sqrt(dot((xs-mx), (xs-mx)) / len(xs)), math.sqrt(dot((ys-my), (ys-my)) / len(ys)))
    #return (mx, my), (50, 50) #(math.sqrt(mean((xs-mx).dot(xs-mx))), math.sqrt(mean((ys-my).dot(ys-my))))

def get_router_levels(list_scan_results):
    router_levels = {}
    for timestamp, scan_results in list_scan_results:
        for scan_result in scan_results:
            if scan_result["BSSID"] in BSSID_TO_ROUTER:
                router = BSSID_TO_ROUTER[scan_result["BSSID"]]
                if not router in router_levels:
                    router_levels[router] = []
                router_levels[router].append(scan_result["level"])

    for router, levels in router_levels.items():
        router_levels[router] = sum(router_levels[router]) / len(router_levels[router])
    return router_levels

def track_location(device_id, timestamp, router_levels=None, scan_results=None):
    global device_samples
    global device_locks

    if not device_id in device_locks:
        with device_locks_lock:
            if not device_id in device_locks:
                device_locks[device_id] = threading.Lock()

    with device_locks[device_id]:
        if scan_results:
            if not device_id in device_scan_results:
                device_scan_results[device_id] = []
            device_scan_results[device_id].append((timestamp, scan_results))
            device_scan_results[device_id] = device_scan_results[device_id][-3:]
            router_levels = get_router_levels(device_scan_results[device_id])

        readings = sorted(router_levels.iteritems(), key=itemgetter(1), reverse=True)
        router_ratios = get_router_distance_ratios(readings)
        router_distances = get_distances_from_readings(readings)

        ratio_model = partial(ratio_observation_probability, router_ratios=router_ratios)
        distance_model = partial(distance_observation_probability, router_distances=router_distances)
        interp_model = partial(interp_observation_probability, router_readings=router_levels)

        def combo_model(xy):
            # logging.info("%.15f, %.15f" % (exp(ratio_model(xy=xy)), COMBO_ALPHA*exp(distance_model(xy=xy))))
            return ratio_model(xy=xy) + COMBO_ALPHA*distance_model(xy=xy)

        observation_models = [combo_model, distance_model, ratio_model, interp_model]

        image_data = []
        device_stats = {}
        for i, model in enumerate(observation_models):
            reweight(device_samples[i][device_id], model)
            #draw_contour(device_samples[device_id])
            device_samples[i][device_id] = resample(device_samples[i][device_id])
            mean_xy, var_xy = get_mean_and_variance(device_samples[i][device_id])
            #image_data.append(draw_image(device_samples[i][device_id]))
            device_samples[i][device_id] = motion(device_samples[i][device_id])

            device_stats[device_id + "_" + str(i)] = {
                    "location" : mean_xy,
                    "variance" : var_xy,
                    "color"    : ["blue", "red", "yellow", "green"][i],
                    }

        # update map information
        update_map_info({
            "info" : \
                "readings      : " + " : ".join(("(%s, %6d)" % x) for x in readings) + "\n" + \
                "router dists  : " + " : ".join(("(%s, %6.3f)" % x) for x in router_distances) + "\n" + \
                "device_stats  : " + str("(%s, %s), var(%s, %s)" % (device_stats[device_id + "_1"]["location"] + device_stats[device_id + "_1"]["variance"])) + "\n" + \
                "",
            "device_stats" : device_stats,
            "images" : image_data,
            })
