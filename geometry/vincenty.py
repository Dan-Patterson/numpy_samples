# -*- coding: UTF-8 -*-
"""
:Script:   vincenty.py
:Author:   Dan.Patterson@carleton.ca
:Created:  2014-10
:Modified: 2017-01-03
:Purpose:
:  Calculates the Vincenty Inverse distance solution for 2 long/lat pairs
:Source:
:  http://www.movable-type.co.uk/scripts/latlong-vincenty.html  java code
: From:
:  T Vincenty, 1975 "Direct and Inverse Solutions of Geodesics on the
:  Ellipsoid with application of nested equations", Survey Review,
:  vol XXIII, no 176, 1975
:  http://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf
:
:Notes:
:  atan2(y,x) or atan2(sin, cos) not like Excel
:  used fmod(x,y) to get the modulous as per python
:
:Returns:
:  distance in meters, initial and final bearings (as an azimuth from N)
:
:Examples:
: long0  lat0  long1  lat1   dist       initial    final  head to
: -75.0, 45.0, -75.0, 46.0   111141.548   0.000,   0.000   N
: -75.0, 46.0, -75.0, 45.0   111141.548 180.000, 180.000   S
: -76.0, 45.0, -75.0, 45.0    78846.334  89.646,  90.353   E
: -75.0, 45.0, -76.0, 45.0    78846.334 270.353, 269.646   W
: -76.0, 46.0, -75.0, 45.0   135869.091 144.526, 145.239   SE
: -75.0, 46.0, -76.0, 45.0   135869.091 215.473, 214.760   SW
: -76.0, 45.0, -75.0, 46.0   135869.091  34.760,  35.473   NE
: -75.0, 45.0, -76.0, 46.0   135869.091 325.239, 324.526   NW
: -90.0,  0.0    0.0   0.0 10018754.171  90.000   90.000   1/4 equator
: -75.0   0.0  -75.0  90.0 10001965.729   0.000    0.000   to N pole
:
:---------------------------------------------------------------------:
"""
#---- imports, formats, constants ----

import sys
import numpy as np
import math
from textwrap import dedent, indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

#---- functions ----


def getDistance(long0, lat0, long1, lat1):
    """return the distance on the ellipsoid between two points using
    :  Vincenty's Inverse method
    : a, b - semi-major and minor axes
    : f - inverse flattening
    : L, dL - delta longitude, initial and subsequent
    : u0, u1 - reduced latitude
    : s_sig - sine sigma
    : c_sig - cosine sigma
    """
    a = 6378137.0
    b = 6356752.314245
    f = 1.0/298.257223563
    twoPI = 2*math.pi
    dL = L = math.radians(long1 - long0)
    u0 = math.atan((1 - f) * math.tan(math.radians(lat0)))
    u1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    s_u0 = math.sin(u0)
    c_u0 = math.cos(u0)
    s_u1 = math.sin(u1)
    c_u1 = math.cos(u1)
    #
    lambdaP  = float()
    iterLimit = 100
    # first approximation
    while (iterLimit > 0):
        sin_dL = math.sin(dL)
        cos_dL = math.cos(dL)
        s_sig = math.sqrt((c_u1*sin_dL)**2 +
                             (c_u0*s_u1 - s_u0*c_u1*cos_dL)**2)     # eq14
        if (s_sig == 0):
            return 0
        c_sig = s_u0*s_u1 + c_u0*c_u1*cos_dL                        # eq 15
        sigma = math.atan2(s_sig, c_sig)                            # eq 16 
        s_alpha = c_u0*c_u1*sin_dL/s_sig                            # eq 17
        cosSqAlpha = 1.0 - s_alpha**2
        if cosSqAlpha != 0.0:
            cos2SigmaM = c_sig - 2.0*s_u0*s_u1/cosSqAlpha           # eq 18
        else:
            cos2SigmaM = c_sig
        C = f/16.0 * cosSqAlpha*(4 + f*(4 - 3*cosSqAlpha))          # eq 10
        lambdaP = dL
        # dL => equation 11
        dL = L + (1 - C)*f*s_alpha*(sigma +
                                    C*s_sig*(cos2SigmaM +
                                             C*c_sig*(-1.0 +
                                                      2*cos2SigmaM**2)))
        #
        if (iterLimit == 0):          #is it time to bail?
            return 0.0
        elif((math.fabs(dL - lambdaP) > 1.0e-12) and (iterLimit > 0)):
            iterLimit = iterLimit - 1
        else:
            break

    uSq = cosSqAlpha * (a**2 - b**2)/b**2
    A = 1 + uSq/16384.0	* (4096 + uSq*(-768 + uSq*(320 - 175*uSq)))  # eq 3
    B = uSq/1024.0 * (256 +  uSq*(-128 + uSq*(74 - 47*uSq)))         # eq 4
    d_sigma = B*s_sig*(cos2SigmaM + 
                (B/4.0)*(c_sig*(-1 + 2*cos2SigmaM**2) - 
                (B/6.0)*cos2SigmaM*(-3 + 4*s_sig**2)*(-3 + 4*cos2SigmaM**2)))
    # d_sigma => eq 6
    dist = b*A*(sigma - d_sigma)                                     # eq 19
    alpha1 = math.atan2(c_u1*sin_dL,  c_u0*s_u1-s_u0*c_u1*cos_dL)
    alpha2 = math.atan2(c_u0*sin_dL, -s_u0*c_u1+c_u0*s_u1*cos_dL)
    # normalize to 0...360  degrees
    alpha1 = math.degrees(math.fmod((alpha1 + twoPI), twoPI))        # eq 20
    alpha2 = math.degrees(math.fmod((alpha2 + twoPI), twoPI))        # eq 21
    return dist, alpha1, alpha2


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    # ----- uncomment one of the  below  -------------------
    coord = [-76.0, 46.0, -75.0, 45.0] # SE
    #coord = [-90.0, 0.0, 0, 0.0]       # 1/4 equator
    #coord = [-75.0, 0.0, -75.0, 90.0]  # to N pole
    a0, a1, a2, a3 = coord
    b0, b1, b2 = getDistance(a0, a1, a2, a3)
    frmt = """
    :--------------------------------------------------------:
    :Vincenty inverse...
    :Longitude, Latitude
    :From: ({:>12.8f}, {:>12.8f})
    :To:   ({:>12.8f}, {:>12.8f})
    :Distance: {:>10.3f} m
    :Bearings...
    :  Initial {:>8.2f} deg
    :  Final   {:>8.2f} deg
    :--------------------------------------------------------:
    """
    print (dedent(frmt).format(a0, a1, a2, a3, b0, b1, b2) )

