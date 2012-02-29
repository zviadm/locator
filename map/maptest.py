import math
import numpy
import operator

PIXELS_PER_METER = 40.0

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

def trilaterate(posA, distA, posB, distB, posC, distC):
    P1 = numpy.array(posA)
    P2 = numpy.array(posB)
    P3 = numpy.array(posC)

    #from wikipedia
    #transform to get circle 1 at origin
    #transform to get circle 2 on x axis
    ex = (P2 - P1)/(numpy.linalg.norm(P2 - P1))
    i = numpy.dot(ex, P3 - P1)
    ey = (P3 - P1 - i*ex)/(numpy.linalg.norm(P3 - P1 - i*ex))
    d = numpy.linalg.norm(P2 - P1)
    j = numpy.dot(ey, P3 - P1)

    #from wikipedia
    #plug and chug using above values
    x = (pow(distA, 2) - pow(distB, 2) + pow(d, 2)) / (2 * d)
    y = ((pow(distA, 2) - pow(distC, 2) + pow(i, 2) + pow(j, 2)) / (2 * j)) - ((i * x) / j)

    #triPt is an array with ECEF x,y of trilateration point
    triPt = P1 + (x * ex) + (y * ey)
    return (int(triPt[0]), int(triPt[1]))

def get_distance_from_level(level):
    # from wikipedia
    level = -level
    if level < 60:
        n = 2.1
    elif level < 80:
        n = 2.5
    else:
        n = 2.9

    C = 20.0 * math.log(4.0 * math.pi / 0.125, 10)
    r_in_meters = 10 ** ((level - C) / (10.0 * n))

    r_in_meters = max(2.5, r_in_meters)
    dist_in_meters = math.sqrt(r_in_meters ** 2 - 2.5 ** 2)
    return dist_in_meters * PIXELS_PER_METER

def get_location(scan_results):
    routers = {}
    for scan_result in scan_results:
        BSSID = scan_result["BSSID"]
        if BSSID in BSSID_TO_ROUTER:
            router = BSSID_TO_ROUTER[BSSID]
            if not router in routers:
                routers[router] = []
            routers[router].append(scan_result["level"])

    sorted_routers = []
    for router, levels in routers.iteritems():
        sorted_routers.append((router, sum(levels) / len(levels)))
    sorted_routers.sort(key = operator.itemgetter(1))

    print "Sorted Routers:", sorted_routers
    assert len(sorted_routers) >= 3

    posA  = ROUTER_POS[sorted_routers[0][0]]
    distA = get_distance_from_level(sorted_routers[0][1])
    posB  = ROUTER_POS[sorted_routers[1][0]]
    distB = get_distance_from_level(sorted_routers[1][1])
    posC  = ROUTER_POS[sorted_routers[2][0]]
    distC = get_distance_from_level(sorted_routers[2][1])

    print trilaterate(posA, distA, posB, distB, posC, distC)

def tstlevels(l1, l2, l3):
    posA  = ROUTER_POS["AP-4-04"]
    distA = get_distance_from_level(l1)
    posB  = ROUTER_POS["AP-4-02"]
    distB = get_distance_from_level(l2)
    posC  = ROUTER_POS["AP-4-05"]
    distC = get_distance_from_level(l3)

    print posA, posB, posC
    print distA, distB, distC
    print trilaterate(posA, distA, posB, distB, posC, distC)
