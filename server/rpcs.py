from __future__ import with_statement, absolute_import

import os

def location_sample(device_id, location_id, timestamp, scan_results):
    for i, scan_result in enumerate(scan_results):
        with open(os.path.join("/srv/locdata", "%s%02d.csv" % (location_id, i)), "a") as f:
            f.write("%s,%s,%s\n" % (scan_result["SSID"], scan_result["BSSID"], scan_result["level"]))


