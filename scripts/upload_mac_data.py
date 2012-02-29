import csv
import re
import requests
import json
import time

"""locator.dropbox.com:80
/rpc
POST
method: 'location_sample'"""



# payload = {
#     'method': 'location_sample',
#     'device_id': '00:00:00:00:00',
#     'location_id': 'awesome_test',
#     'timestamp': '1',
#     'scan_results': [
#       {
#         'SSID': 'Dropbox',
#         'BSSID': '0a:00:00:00:00',
#         'level': '-80',
#       },
#       {
#         'SSID': 'DropboxGuest',
#         'BSSID': '0b:00:00:00:00',
#         'level': '-81',
#       },
#       {
#         'SSID': 'DropboxFoo',
#         'BSSID': '0c:00:00:00:00',
#         'level': '-82',
#       },
#       ]
#     }
# print json.dumps(payload)

# print "making request"
# r = requests.post("http://locator.dropbox.com:80/rpc",
#                   data=json.dumps(payload))
# print "okay"


def location_sample_rpc(payload):
    payload['method'] = 'location_sample'
    r = requests.post("http://locator.dropbox.com:80/rpc",
                      data=json.dumps(payload))
    return r.text


DATA = [
  '../data/bathroom.csv',
  '../data/conferenceroom.csv',
  '../data/doorman.csv',
  '../data/jiesdesk.csv',
  '../data/kitchen.csv',
  '../data/mobile.csv',
  '../data/standing_bathroom1.csv',
  '../data/standing_bathroom2.csv',
  '../data/standing_conferenceroom1.csv',
  '../data/standing_conferenceroom2.csv',
  '../data/standing_doorman1.csv',
  '../data/standing_doorman2.csv',
  '../data/standing_jiesdesk1.csv',
  '../data/standing_jiesdesk2.csv',
  '../data/standing_kitchen1.csv',
  '../data/standing_kitchen2.csv',
  '../data/standing_mobile1.csv',
  '../data/standing_mobile2.csv',
  "../data/walking_bathroom1.csv",
  "../data/walking_bathroom10.csv",
  "../data/walking_bathroom2.csv",
  "../data/walking_bathroom3.csv",
  "../data/walking_bathroom4.csv",
  "../data/walking_bathroom5.csv",
  "../data/walking_bathroom6.csv",
  "../data/walking_bathroom7.csv",
  "../data/walking_bathroom8.csv",
  "../data/walking_bathroom9.csv",
  "../data/walking_doorman1.csv",
  "../data/walking_doorman10.csv",
  "../data/walking_doorman2.csv",
  "../data/walking_doorman3.csv",
  "../data/walking_doorman4.csv",
  "../data/walking_doorman5.csv",
  "../data/walking_doorman6.csv",
  "../data/walking_doorman7.csv",
  "../data/walking_doorman8.csv",
  "../data/walking_doorman9.csv",
  "../data/walking_jiesdesk1.csv",
  "../data/walking_jiesdesk10.csv",
  "../data/walking_jiesdesk2.csv",
  "../data/walking_jiesdesk3.csv",
  "../data/walking_jiesdesk4.csv",
  "../data/walking_jiesdesk5.csv",
  "../data/walking_jiesdesk6.csv",
  "../data/walking_jiesdesk7.csv",
  "../data/walking_jiesdesk8.csv",
  "../data/walking_jiesdesk9.csv",
  "../data/walking_kitchen1.csv",
  "../data/walking_kitchen10.csv",
  "../data/walking_kitchen2.csv",
  "../data/walking_kitchen3.csv",
  "../data/walking_kitchen4.csv",
  "../data/walking_kitchen5.csv",
  "../data/walking_kitchen6.csv",
  "../data/walking_kitchen7.csv",
  "../data/walking_kitchen8.csv",
  "../data/walking_kitchen9.csv",
  "../data/walking_mobile1.csv",
  "../data/walking_mobile10.csv",
  "../data/walking_mobile2.csv",
  "../data/walking_mobile3.csv",
  "../data/walking_mobile4.csv",
  "../data/walking_mobile5.csv",
  "../data/walking_mobile6.csv",
  "../data/walking_mobile7.csv",
  "../data/walking_mobile8.csv",
  "../data/walking_mobile9.csv",
]


train_label_matcher = re.compile(".*/(.*?)\d*\.csv")

for t in DATA:
    location_id = train_label_matcher.match(t).groups(1)[0]
    with open(t, 'rb') as f:
        reader = csv.reader(f)
        payload = {
            'device_id': 'e4:ce:8f:30:2d:a8',
            'location_id': location_id,
            'timestamp': str(time.time()),
            'scan_results': [
            {'SSID': ssid,
             'BSSID': bssid,
             'level': level,
             } for ssid, bssid, level in reader
            ]
            }
        # print payload
        location_sample_rpc(payload)
    print t
    


