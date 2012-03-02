from __future__ import with_statement, absolute_import
import os
import threading
import time

import tracker

def location_sample(device_id, location_id, timestamp, scan_results):
    LOCDATA_DIR = "/srv/locdata"

    max_index = 0
    csvfiles = os.listdir(LOCDATA_DIR)
    for csvfile in csvfiles:
        if csvfile.startswith(location_id):
            try:
                index = int(csvfile[len(location_id):-4])
                if index > max_index:
                    max_index = index
            except ValueError:
                pass

    with open(os.path.join("/srv/locdata", "%s%02d.csv" % (location_id, max_index + 1)), "w") as f:
        for scan_result in scan_results:
            f.write("%s,%s,%s\n" % (scan_result["SSID"], scan_result["BSSID"], scan_result["level"]))
    return { "ret" : "ok" }

def track_location(device_id, timestamp, router_levels):
    tracker.track_location(device_id, timestamp, router_levels)
    return { "ret" : "ok" }

def get_locations():
    return {
        "locations" : {
            "c8:aa:21:b0:53:c3" : (1000 + (int(time.time()) % 20) * 80, 200 + (int(time.time()) % 20) * 30),
            }
        }
