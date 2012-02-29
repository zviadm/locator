import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import pickle
with open('cached_kde_map', 'rb') as f:
    cache = pickle.load(f)

device_id = cache.keys()[0]
print device_id
# print set([x for x in cache[device_id].values()])

img = mpimg.imread('../images/corner.png')
imgplot = plt.imshow(img)


sampling_locs = [
    (x, y) for x in range(140, 250, 5) for y in range(10, 170, 5)
    ]

Rs = range(-30, -81, 1)

bests = [[min((cache[device_id][(x, y, R)],R) for R in range(-30, -81, -1))[1]
          for x in range(140, 250, 5)]
         for y in range(10, 170, 5)
         ]

# bests = [[cache[device_id][(x, y, -30)]
#           for x in range(140, 250, 5)]
#          for y in range(10, 170, 5)
#          ]

print bests[0]
# Zs = []
# for x in range(140, 250, 5):
#     print x
#     for y in range(10, 170, 5):
#         Zs.append((x, y, min(cache[device_id][(x, y, R)] for R in range(-30, 


# x, y = zip(*sampling_locs)
# plt.plot(x, y, 'o', markersize=2)

print len(range(140, 250, 5)), len(range(10, 170, 5))
print len(bests), len(bests[0])


CS = plt.contour(range(140, 250, 5), range(10, 170, 5), bests)
cbar = plt.colorbar(CS)

plt.savefig('test.png')
