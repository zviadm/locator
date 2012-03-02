from __future__ import with_statement, absolute_import
import os
import threading
import time

import tracker
from tracker_info import get_device_stats

def location_sample(device_id, location_id, timestamp, scan_results):
    LOCDATA_DIR = "/srv/locdata"

    with open(os.path.join(LOCDATA_DIR, "%s.csv" % (location_id)), "a") as f:
        for scan_result in scan_results:
            f.write("%s,%s,%s,%s,%s,%s\n" % (device_id, location_id, timestamp, scan_result["SSID"], scan_result["BSSID"], scan_result["level"]))
    return { "ret" : "ok" }

def track_location(device_id, timestamp, router_levels):
    tracker.track_location(device_id, timestamp, router_levels)
    return { "ret" : "ok" }

def get_locations():
    return {
        "devices" : get_device_stats(),
        }
