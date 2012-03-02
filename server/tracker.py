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
INTERP_STD_OFFSET = 1.0
MAX_PARTICLES = 400

RUNNING_AVERAGE_LENGTH = 3

# physical constants for determining path loss (wikipedia)
WAVELENGTH = 0.125
# N_COEFFS = [-0.07192023, -2.40415772]
N_COEFFS = [-0.07363796, -2.52218124]

# map constants
ROUTER_HEIGHT = 2.1
PIXELS_PER_METER = 22.0
PIXELS_PER_METER_SQ = PIXELS_PER_METER**2
POWERFUL_ROUTERS = set(['AP-1-07', 'AP-4-02'])
CHANNEL_CORRECTION = 1.8

ROUTER_POS = {
        # Part 1
        "AP-1-01" : (447, 254),
        "AP-1-02" : (621, 171),
        "AP-1-03" : (563, 397),
        "AP-1-04" : (613, 613),
        "AP-1-05" : (975, 207),
        "AP-1-06" : (938, 610),
        "AP-1-07" : (1507, 149),
        "AP-1-08" : (1474, 601),

        # Part 2
        "AP-2-01" : (1674, 391),
        "AP-2-02" : (1859, 641),
        "AP-2-03" : (1858, 167),
        "AP-2-04" : (2153, 383),
        "AP-2-05" : (2422, 179),
        "AP-2-06" : (2482, 630),
        "AP-2-07" : (2882, 157),
        "AP-2-08" : (2877, 656),
        "AP-2-09" : (3136, 559),
        "AP-2-10" : (3348, 175),

        # Part 3
        "AP-3-01" : (3741, 311),
        "AP-3-02" : (3913, 163),
        "AP-3-03" : (3957, 629),
        "AP-3-04" : (4133, 398),
        "AP-3-05" : (4505, 171),
        "AP-3-06" : (4515, 621),

        # Part 4
        "AP-4-01" : (4980, 600),
        "AP-4-02" : (4980, 200), 
        "AP-4-03" : (5041, 409),
        "AP-4-04" : (5325, 159),
        "AP-4-05" : (5265, 537),
        }

BSSID_TO_ROUTER = {
        # Part 1
        "00:0b:86:74:95:90" : "AP-1-01",
        "00:0b:86:74:95:91" : "AP-1-01",
        "00:0b:86:74:90:90" : "AP-1-02",
        "00:0b:86:74:90:91" : "AP-1-02",
        # "00:0b:86:74:97:f0" : "AP-1-03",
        # "00:0b:86:74:97:f1" : "AP-1-03",
        "00:0b:86:74:90:80" : "AP-1-04",
        "00:0b:86:74:90:81" : "AP-1-04",
        "00:0b:86:74:8d:90" : "AP-1-05",
        "00:0b:86:74:8d:91" : "AP-1-05",
        "00:0b:86:74:90:60" : "AP-1-06",
        "00:0b:86:74:90:61" : "AP-1-06",
        "00:0b:86:74:98:30" : "AP-1-07",
        "00:0b:86:74:98:31" : "AP-1-07",
        "00:0b:86:74:8f:40" : "AP-1-08",
        "00:0b:86:74:8f:41" : "AP-1-08",

        # Part 2
        "00:0b:86:74:8f:90" : "AP-2-01",
        "00:0b:86:74:8f:91" : "AP-2-01",
        "00:0b:86:74:99:e0" : "AP-2-02",
        "00:0b:86:74:99:e1" : "AP-2-02",
        "00:0b:86:74:8f:70" : "AP-2-03",
        "00:0b:86:74:8f:71" : "AP-2-03",
        # "00:0b:86:74:97:80" : "AP-2-04",
        # "00:0b:86:74:97:81" : "AP-2-04",
        "00:0b:86:74:90:00" : "AP-2-05",
        "00:0b:86:74:90:01" : "AP-2-05",
        "00:0b:86:74:95:b0" : "AP-2-06",
        "00:0b:86:74:95:b1" : "AP-2-06",
        "00:0b:86:74:98:10" : "AP-2-07",
        "00:0b:86:74:98:11" : "AP-2-07",
        "00:0b:86:74:9a:00" : "AP-2-08",
        "00:0b:86:74:9a:01" : "AP-2-08",
        "00:0b:86:74:98:40" : "AP-2-09",
        "00:0b:86:74:98:41" : "AP-2-09",
        "00:0b:86:74:90:50" : "AP-2-10",
        "00:0b:86:74:90:51" : "AP-2-10",

        # Part 3
        #"00:0b:86:74:90:f0" : "AP-3-01",
        #"00:0b:86:74:90:f1" : "AP-3-01",
        "00:0b:86:74:95:e0" : "AP-3-02",
        "00:0b:86:74:95:e1" : "AP-3-02",
        "00:0b:86:74:9a:20" : "AP-3-03",
        "00:0b:86:74:9a:21" : "AP-3-03",
        #"00:0b:86:74:99:a0" : "AP-3-04",
        #"00:0b:86:74:99:a1" : "AP-3-04",
        "00:0b:86:74:99:b0" : "AP-3-05",
        "00:0b:86:74:99:b1" : "AP-3-05",
        "00:0b:86:74:95:a0" : "AP-3-06",
        "00:0b:86:74:95:a1" : "AP-3-06",

        # Part 4
        "00:0b:86:74:96:80" : "AP-4-01",
        "00:0b:86:74:96:81" : "AP-4-01",
        "00:0b:86:74:97:60" : "AP-4-02",
        "00:0b:86:74:97:61" : "AP-4-02",
        # "00:0b:86:74:9a:80" : "AP-4-03",
        # "00:0b:86:74:9a:81" : "AP-4-03",
        "00:0b:86:74:9a:90" : "AP-4-04",
        "00:0b:86:74:9a:91" : "AP-4-04",
        "00:0b:86:74:97:90" : "AP-4-05",
        "00:0b:86:74:97:91" : "AP-4-05",
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
    return [(ROUTER_POS[r], get_distance_from_level(l)*(CHANNEL_CORRECTION if r in POWERFUL_ROUTERS else 1.0)) for r, l in router_readings if l > -80]


NORM_Z = log(0.39894)
def loglikelihood(x):
    return NORM_Z - 0.5*x*x

def distance_observation_probability(router_distances, xy):
    ll = 0
    x, y = xy

    for (x1, y1), distance in router_distances:
        dist = math.sqrt((x - x1)**2/PIXELS_PER_METER_SQ + (y - y1)**2 / PIXELS_PER_METER_SQ)
        ll += max(LOG_MIN_PROB, loglikelihood((distance - dist) / max(MIN_DISTANCE_STD, distance**2/25.0)))

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

# model_data: location: {normed_device_id: (mean, variance)}
TRAINING_DATA = [
    # ('../locdata/zviad1.csv', 'zviad1', (5295, 196)), #z1
    # ('../locdata/zviad2.csv', 'zviad2', (5295, 353)), #z2
    # ('../locdata/zviad3.csv', 'zviad3', (5285, 554)), #z3
    # ('../locdata/zviad4.csv', 'zviad4', (5029, 544)), #z4
    # ('../locdata/zviad5.csv', 'zviad5', (5029, 396)), #z5
    # ('../locdata/zviad6.csv', 'zviad6', (5032, 237)), #z6
    # ('../locdata/zviad7.csv', 'zviad7', (5478, 152)), #z7
    ('../locdata/ft1.csv',  'ft1',  (5095, 83)),
    ('../locdata/ft2.csv',  'ft2',  (5178, 200)),
    ('../locdata/ft3.csv',  'ft3',  (5408, 258)),
    ('../locdata/ft4.csv',  'ft4',  (5425, 477)),
    ('../locdata/ft5.csv',  'ft5',  (5170, 540)),
    ('../locdata/ft6.csv',  'ft6',  (5095, 691)),
    ('../locdata/ft7.csv',  'ft7',  (4864, 536)),
    ('../locdata/ft8.csv',  'ft8',  (4829, 235)),
    ('../locdata/ft9.csv',  'ft9',  (5138, 453)),
    ('../locdata/ft10.csv', 'ft10', (5138, 342)),
    ('../locdata/ft11.csv', 'ft11', (5032, 459)),
    ('../locdata/ft13.csv', 'ft13', (5288, 420)),
    ('../locdata/ft14.csv', 'ft14', (5032, 331)),
    ]

def build_model(training_data=TRAINING_DATA):
    model_data = {}
    for training_fname, label, location in training_data:
        data = defaultdict(list)
        with open(training_fname, 'rb') as f:
          reader = csv.reader(f)
          for recording_id, loc_name, ts, ssid, device_id, strength in reader:
            if True or 'Dropbox' in ssid:
                data[device_id].append(float(strength))
                # m = normalize_mac(device_id)
                # if m in BSSID_TO_ROUTER:
                #     data[BSSID_TO_ROUTER[m]].append(float(strength))
        # NSAMPLES=10
        model_data[location] = dict((key, (mean(val), INTERP_STD_OFFSET+std(val))) for key, val in data.iteritems() if len(val) > 30 and mean(val) > -80)
    return model_data

model_data = build_model()

def distancesq(xy1, xy2):
    x1, y1 = xy1
    x2, y2 = xy2
    return (x1-x2)**2 + (y1-y2)**2

def interp_observation_probability(model, router_readings, xy):
    (dist1sq, loc1, l1), (dist2sq, loc2, l2) = sorted((distancesq(xy, location), device_dict, location) for location, device_dict in model.iteritems())[:2]

    # print "closest locs: ", dist1sq, l1
    # print "closest locs: ", dist2sq, l2

    dist1 = math.sqrt(dist1sq)
    dist2 = math.sqrt(dist2sq)

    alpha = dist2/(dist1+dist2)

    ll = 0.0
    couldnt = 0
    for entry_dict in router_readings:
        device = entry_dict['BSSID']
        signal = float(entry_dict['level'])
        if signal < -85: continue
        if device in loc1 and device in loc2:
            # p1 = loglikelihood((signal - loc1[device][0])/(loc1[device][1]))
            # p2 = loglikelihood((signal - loc2[device][0])/(loc2[device][1]))
            # ll += max(LOG_MIN_PROB, alpha*p1 + (1-alpha)*p2)
            mu = loc1[device][0] * alpha + loc2[device][0] * (1-alpha)
            sigma = loc1[device][1] * alpha + loc2[device][1] * (1-alpha)
            ll += max(LOG_MIN_PROB, loglikelihood((signal - mu) / sigma))
        elif device in loc1:
            ll += max(LOG_MIN_PROB, loglikelihood((signal - loc1[device][0]) / loc1[device][1]))
        elif device in loc2:
            ll += max(LOG_MIN_PROB, loglikelihood((signal - loc2[device][0]) / loc2[device][1]))
        else:
            ll += LOG_MIN_PROB
    # logging.info("couldnt: (%d / %d)" % (couldnt, len(router_readings)))
    return ll


# obs reweight
def reweight(samples, observation_model):
    Z = 0.0
    for i in range(len(samples)):
        weight, xy = samples[i]
        nw = weight * 10**observation_model(xy=xy)
        samples[i][0] = nw
        Z += nw

    if Z == 0.0:
        return

    for i in range(len(samples)):
        samples[i][0] /= Z


# resample
def resample(samples):
    counts = random.multinomial(MAX_PARTICLES, zip(*samples)[0])
    return [list(x) for x in zip(counts, zip(*samples)[1]) if x[0] > 0]

def motion(samples):
    toret = []
    for i in range(len(samples)):
        w, (x, y) = samples[i]
        for j in xrange(w):
            toret.append([1, (x + random.normal(0, MOTION_STDEV), y + random.normal(0, MOTION_STDEV))])

    for pos in ROUTER_POS.values():
        toret.append([1, (x, y)])
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
        router_levels[router] = float(sum(router_levels[router])) / len(router_levels[router])
    return router_levels

UPDATE_ALPHA = 0.3
def update_scan_results(new_scan_results, device_id):
    toret = {}
    for scan_result in new_scan_results:
        bssid, level = scan_result["BSSID"], scan_result["level"]
        if bssid in BSSID_TO_ROUTER:
            router = BSSID_TO_ROUTER[bssid]
            if not router in device_scan_results[device_id]:
                device_scan_results[device_id][router] = float(level)
            else:
                device_scan_results[device_id][router] = device_scan_results[device_id][router]*(1-UPDATE_ALPHA) + float(level) * (UPDATE_ALPHA)
            toret[router] = device_scan_results[device_id][router]

    return toret

def track_location(device_id, timestamp, router_levels=None, scan_results=None):
    global device_samples
    global device_locks

    if not device_id in device_locks:
        with device_locks_lock:
            if not device_id in device_locks:
                device_locks[device_id] = threading.Lock()

    with device_locks[device_id]:
        # if scan_results:
        #     if not device_id in device_scan_results:
        #         device_scan_results[device_id] = []
        #     device_scan_results[device_id].append((timestamp, scan_results))
        #     device_scan_results[device_id] = device_scan_results[device_id][-RUNNING_AVERAGE_LENGTH:]
        #     router_levels = get_router_levels(device_scan_results[device_id])

        if scan_results:
            if not device_id in device_scan_results:
                device_scan_results[device_id] = {}
            router_levels = update_scan_results(scan_results, device_id)

        readings = sorted(router_levels.iteritems(), key=itemgetter(1), reverse=True)
        router_ratios = get_router_distance_ratios(readings)
        router_distances = get_distances_from_readings(readings)

        ratio_model = partial(ratio_observation_probability, router_ratios=router_ratios)
        distance_model = partial(distance_observation_probability, router_distances=router_distances)

        interp_model = partial(interp_observation_probability, router_readings=scan_results, model=model_data)

        def combo_model(xy):
            # logging.info("%.15f, %.15f" % (exp(ratio_model(xy=xy)), COMBO_ALPHA*exp(distance_model(xy=xy))))
            return ratio_model(xy=xy) + COMBO_ALPHA*distance_model(xy=xy)

        #observation_models = [combo_model, distance_model, ratio_model, interp_model]
        observation_models = [interp_model, distance_model]

        image_data = []
        device_stats = {}
        for i, model in enumerate(observation_models):
            reweight(device_samples[i][device_id], model)
            #draw_contour(device_samples[device_id])
            device_samples[i][device_id] = resample(device_samples[i][device_id])
            if len(device_samples[i][device_id]) == 0:
                device_samples[i][device_id] = [[1.0, (x, y)] for x in range(XMIN, XMAX, XSTEP) for y in range(YMIN, YMAX, YSTEP)]

            mean_xy, var_xy = get_mean_and_variance(device_samples[i][device_id])
            #image_data.append(draw_image(device_samples[i][device_id]))
            device_samples[i][device_id] = motion(device_samples[i][device_id])

            device_stats[device_id + "_" + str(i)] = {
                    "location" : mean_xy,
                    "variance" : var_xy,
                    "color"    : ["green", "red"][i],
                    }

        # update map information
        update_map_info({
            "info" : \
                "readings      : " + " : ".join(("(%s, %6.3f)" % x) for x in readings) + "\n" + \
                "router dists  : " + " : ".join(("(%s, %6.3f)" % x) for x in router_distances) + "\n" + \
                "device_stats  : " + str("(%s, %s), var(%s, %s)" % (device_stats[device_id + "_0"]["location"] + device_stats[device_id + "_0"]["variance"])) + "\n" + \
                "device_stats  : " + str("(%s, %s), var(%s, %s)" % (device_stats[device_id + "_1"]["location"] + device_stats[device_id + "_1"]["variance"])) + "\n" + \
                "",
            "device_stats" : device_stats,
            "images" : image_data,
            })
