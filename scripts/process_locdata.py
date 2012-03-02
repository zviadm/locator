import math
import csv
from collections import defaultdict

LOCATIONS = {
    'zviad1': (5209, 186),
    'zviad2': (5210, 354),
    'zviad3': (5202, 553),
    'zviad4': (4948, 546),
    'zviad5': (4953, 398),
    'zviad6': (4949, 242),
    'zviad7': (5396, 152),
    }

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
        "AP-4-01" : (4897, 598),
        "AP-4-02" : (4898, 200),
        "AP-4-03" : (4958, 412),
        "AP-4-04" : (5242, 158),
        "AP-4-05" : (5182, 538),
        }


DATA = [
    '../locdata/zviad1.csv',
    '../locdata/zviad2.csv',
    '../locdata/zviad3.csv',
    '../locdata/zviad4.csv',
    '../locdata/zviad5.csv',
    '../locdata/zviad6.csv',
    '../locdata/zviad7.csv',
    ]

PIXEL_TO_M = 22

def find_distance(router_loc, reading_loc):
    x1, y1 = router_loc

    x2, y2 = reading_loc

    return math.sqrt((x1-x2)**2/PIXEL_TO_M**2 + (y1 - y2)**2/PIXEL_TO_M**2. + 2.1**2)

output = []
router_specific = defaultdict(list)
location_specific = defaultdict(list)
for fname in DATA:
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for device_id, loc_id, ts, ssid, bssid, signal in reader:
            if bssid in BSSID_TO_ROUTER:
                router_loc = ROUTER_POS[BSSID_TO_ROUTER[bssid]]
                reading_loc = LOCATIONS[loc_id]
     
                distance = find_distance(router_loc, reading_loc)
                output.append((distance, float(signal)))
                router_specific[BSSID_TO_ROUTER[bssid]].append((distance, float(signal)))
                location_specific[fname].append((distance, float(signal)))

with open('../locdata/all_zviad.csv', 'wb') as fout:
    writer = csv.writer(fout)
    for entry in output:
        writer.writerow(entry)

for router_name, entries in router_specific.iteritems():
    with open('../locdata/router/%s.csv' % router_name[-2:], 'wb') as fout:
        writer = csv.writer(fout)
        for entry in entries:
            writer.writerow(entry)

for fname, entries in location_specific.iteritems():
    with open('../locdata/location/%s.csv' % fname[-5:], 'wb') as fout:
        writer = csv.writer(fout)
        for entry in entries:
            writer.writerow(entry)

