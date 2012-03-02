import csv
import re
import sys

from collections import defaultdict

from numpy import *
from scipy import stats

test_label_matcher = re.compile(".*?_(.*?)\d*\.csv")
train_label_matcher = re.compile(".*/(.*?)\d*\.csv")
# train_label_matcher = re.compile(".*/(.*?)\.csv")


fname = '../data/conference_room.csv'
if len(sys.argv) > 1:
  fname = sys.argv[1]


# SIGNAL_STRENGTH_THRESH = -75
# LOG_MIN_PROB = -200
SIGNAL_STRENGTH_THRESH = -75
LOG_MIN_PROB = -50

class Location(object):
  def __init__(self, fname):
    self.data = defaultdict(list)
    with open(fname, 'rb') as f:
      reader = csv.reader(f)
      for ssid, device_id, strength in reader:
        if True or 'Dropbox' in ssid:
          self.data[device_id].append(float(strength))
    
    self.dists = {}
    for device, readings in self.data.iteritems():
      if all([x > SIGNAL_STRENGTH_THRESH for x in readings]):
        # if std(readings) == 0:
        #   print device, mean(readings), std(readings), max(readings), min(readings), len(readings)
        # self.dists[device] = stats.norm(mean(readings), max(0.2, std(readings)))
        self.dists[device] = (mean(readings), max(0.2, std(readings)))

  def logprob(self, readings):
    all_devices = set(readings.keys()).union(set(self.dists.keys()))

    ll = 0.0
    for device in all_devices:
      if device in readings and device in self.dists:
        ll += max(LOG_MIN_PROB, loglikelihood((float(readings[device]) - self.dists[device][0])/self.dists[device][1])
        # ll += log(max(exp(LOG_MIN_PROB), self.dists[device].pdf(float(readings[device]))))
      elif device in readings and readings[device] > SIGNAL_STRENGTH_THRESH: # in readings not in model
        ll += LOG_MIN_PROB
      elif device in self.dists: # in model not in readings
        ll += LOG_MIN_PROB
    return ll

TRAINING_DATA = [
  '../data/bathroom.csv',
  '../data/conferenceroom.csv',
  '../data/doorman.csv',
  '../data/jiesdesk.csv',
  '../data/kitchen.csv',
  '../data/mobile.csv',
  ]

TRAINING_DATA = [
  '../data/bathroom.csv',
  '../data/conferenceroom.csv',
  '../data/doorman.csv',
  '../data/jiesdesk.csv',
  '../data/kitchen.csv',
  '../data/mobile.csv',
  "../newdata/arrears1.csv",
  "../newdata/breakfastbar1.csv",
  "../newdata/breakuproom.csv",
  "../newdata/bromancechamber1.csv",
  "../newdata/client1.csv",
  "../newdata/drew1.csv",
  "../newdata/molly1.csv",
  ]


class NaiveBayesModel(object):
  def __init__(self, training_fnames):
    self.models = {}
    for fname in training_fnames:
      train_label = train_label_matcher.match(fname).groups(1)[0]
      self.models[train_label] = Location(fname)

  def get_likelihoods(self, readings):
    results = []
    for fname, model in self.models.iteritems():
      results.append((model.logprob(reading), fname))
    return sorted(results, reverse=True)

  def get_location(self, readings):
    return max((model.logprob(readings), fname) for fname, model in self.models.iteritems())

    
model = NaiveBayesModel(TRAINING_DATA)

TEST_DATA = ['../data/standing_bathroom1.csv',
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
             ]

TEST_DATA2 = [
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

TEST_DATA = [
  "../newdata/standing_arrears1.csv",
  "../newdata/standing_arrears2.csv",
  "../newdata/standing_breakfastbar1.csv",
  "../newdata/standing_breakfastbar2.csv",
  "../newdata/standing_breakuproom1.csv",
  "../newdata/standing_breakuproom2.csv",
  "../newdata/standing_client1.csv",
  "../newdata/standing_client2.csv",
  "../newdata/standing_drew1.csv",
  "../newdata/standing_drew2.csv",
  "../newdata/standing_molly1.csv",
  "../newdata/standing_molly2.csv",
]

testdata = {}
for fname in TEST_DATA:
  toeval = {}
  with open(fname, 'rb') as fin:
    reader = csv.reader(fin)
    for s, device, strength in reader:
      if True or 'Dropbox' in s:
        toeval[device] = strength 
  testdata[fname] = toeval



correct = 0
for fname, reading in sorted(testdata.iteritems()):
  test_label = test_label_matcher.match(fname).groups(1)[0]

  loc = model.get_location(reading)

  if test_label == loc[1]:
    correct += 1
  else:
    print
    print fname, test_label, loc[1]
    print " ".join("%s: %.2f" % (label, prob) for prob, label in model.get_likelihoods(reading))
    print

print "Accuracy: %.2f (%d / %d)" % (float(correct) / len(testdata), correct, len(testdata))
  
