import math
import csv

LOCATIONS = {
    'zviad1': (1, 1),
    'zviad2': (1, 1),
    'zviad3': (1, 1),
    'zviad4': (1, 1),
    'zviad5': (1, 1),
    'zviad6': (1, 1),    
    'zviad7': (1, 1),    
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
        "AP-4-01" : (500, 1105),
        "AP-4-02" : (450, 292),
        "AP-4-03" : (687, 724),
        "AP-4-04" : (1203, 315),
        "AP-4-05" : (1130, 970),
        }



DATA = [
    '../locdata/zviad1.csv',
    '../locdata/zviad2.csv',
    '../locdata/zviad3.csv',
    '../locdata/zviad4.csv',
    '../locdata/zviad5.csv',
    '../locdata/zviad6.csv',
    '../locdata/zviad7.csv',
    '../locdata/zviad1.csv',
    ]

def find_distance(router_loc, reading_loc):
    x1, y1 = router_loc
    x2, y2 = reading_loc

    return math.sqrt((x1-x2)**2 + (y1 - y2)**2 + 2.5**2)

output = []
for fname in DATA:
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for device_id, loc_id, ts, ssid, bssid, signal in reader:
            if bssid in BSSID_TO_ROUTER:
                router_loc = ROUTER_POS[BSSID_TO_ROUTER[bssid]]
                reading_loc = LOCATIONS[loc_id]
     
                distance = find_distance(router_loc, reading_loc)
                output.append((distance, float(signal)))

with open('../locdata/all_zviad.csv', 'wb') as fout:
    writer = csv.writer(fout)
    for entry in output:
        writer.writerow(entry)
