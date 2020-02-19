from dists.dist import dist, neighbors
import numpy as np
import time


def dist_segments(p1, p2, p3, p4):
    '''
    translated for matlab
    https://www.mathworks.com/matlabcentral/fileexchange/32487-shortest-distance-between-two-line-segments
    by yohai magen 12-2019
    '''
    u = p1 - p2
    v = p3 - p4
    w = p2 - p4

    a = u.dot(u)
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w)
    e = v.dot(w)
    D = a * c - b * b
    sD = D
    tD = D

    epsilon = 1e-8
    if D < epsilon:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if (sN < 0.0):
            sN = 0.0
            tN = e
            tD = c
        elif (sN > sD):
            sN = sD
            tN = e + b
            tD = c
    if (tN < 0.0):
        tN = 0.0
        if (-d < 0.0):
            sN = 0.0
        elif (-d > a):
            sN = sD
        else:
            sN = -d
            sD = a
    elif (tN > tD):
        tN = tD
        if ((-d + b) < 0.0):
            sN = 0.0
        elif ((-d + b) > a):
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    if (np.abs(sN) < epsilon):
        sc = 0.0
    else:
        sc = sN / sD
    if (np.abs(tN) < epsilon):
        tc = 0.0
    else:
        tc = tN / tD

    dP = w + (sc * u) - (tc * v)

    return np.linalg.norm(dP)

ps = np.random.uniform(0, 100, (4, 3))

p1 = np.array([34.23459325, 65.54939893,  5.        ])
p2 = np.array([37.64458505, 61.89263042,  5.        ])
p3 = np.array([37.64458505, 61.89263042, 10.        ])
p4 = np.array([34.23459325, 65.54939893, 10.        ])
########
t1 = np.array([32.31420869, 68.73983216,  0.        ])
t2 = np.array([35.59450384, 64.96628426,  0.        ])
t3 = np.array([35.20006068, 64.62340005,  4.97260948])
t4 = np.array([31.91976553, 68.39694795,  4.97260948])

ps = (p1, p2, p3, p4)
ts = (t1, t2, t3, t4)

for i in range(4):
    for j in range(4):
        print(dist_segments(ps[i], ps[(i + 1) % 4], ts[j], ts[(j + 1) % 4]))
        print(dist(ps[i], ps[(i + 1) % 4], ts[j], ts[(j + 1) % 4]))
        print('#####')

print(neighbors(p1, p2, p3, p4, t1, t2, t3, t4))