from __future__ import with_statement, absolute_import

import os

def location_sample(device_id, location_id, timestamp, scan_results):
    with open(os.path.join("/srv/locdata", location_id + ".csv"), "a") as f:
        for scan_result in scan_results:
            f.write("%s,%s,%s\n" % (scan_result["SSID"], scan_result["BSSID"], scan_result["level"]))


