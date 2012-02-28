#!/usr/bin/env python
from subprocess import Popen, PIPE
from plistlib import readPlist
from pprint import pprint
import sys

AirportPath = '/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport'

def scan_networks():
  proc = Popen([AirportPath, '-s', '-x'], stdout=PIPE)
  ssid_data = readPlist(proc.stdout)

  return [(x['SSID_STR'], x['BSSID'], x['RSSI']) for x in ssid_data]


import csv

fname = 'testreading'
nsamples = 10
nbatches = 1
if len(sys.argv) > 1:
  fname = sys.argv[1]
if len(sys.argv) > 2:
  nsamples = int(sys.argv[2])
if len(sys.argv) > 3:
  nbatches = int(sys.argv[3])

for batch in range(nbatches):
  with open("%s%d.csv" % (fname, batch+1), 'wb') as fout:
    writer = csv.writer(fout)
    for i in range(nsamples):
      print "batch %d/%d sample %d/%d" % (batch+1, nbatches, i+1, nsamples)
      writer.writerows(scan_networks())
