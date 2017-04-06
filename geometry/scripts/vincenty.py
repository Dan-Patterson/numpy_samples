# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import math
import numpy as np


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def geodesic(long0, lat0, long1, lat1, verbose=False):
    """return the distance on the ellipsoid between two points using
    :  Vincenty's Inverse method
    : a, b - semi-major and minor axes WGS84 model
    : f - inverse flattening
    : L, dL - delta longitude, initial and subsequent
    : u0, u1 - reduced latitude
    : s_sig - sine sigma
    : c_sig - cosine sigma
    """
    a = 6378137.0
    b = 6356752.314245
    ab_b = (a**2 - b**2)/b**2
    f = 1.0/298.257223563
    twoPI = 2*math.pi
    dL = L = math.radians(long1 - long0)
    u0 = math.atan((1 - f) * math.tan(math.radians(lat0)))  # reduced latitudes
    u1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    s_u0 = math.sin(u0)
    c_u0 = math.cos(u0)
    s_u1 = math.sin(u1)
    c_u1 = math.cos(u1)
    # ---- combine repetitive terms ----
    sc_01 = s_u0*c_u1
    cs_01 = c_u0*s_u1
    cc_01 = c_u0*c_u1
    ss_01 = s_u0*s_u1
    #
    lambdaP = float()
    max_iter = 20
    # first approximation
    cnt = 0
    while (cnt < max_iter):
        s_dL = math.sin(dL)
        c_dL = math.cos(dL)
        s_sig = math.sqrt((c_u1*s_dL)**2 + (cs_01 - sc_01*c_dL)**2)  # eq14
        if (s_sig == 0):
            return 0
        c_sig = ss_01 + cc_01*c_dL                      # eq 15
        sigma = math.atan2(s_sig, c_sig)                # eq 16
        s_alpha = cc_01*s_dL/s_sig                      # eq 17
        c_alpha2 = 1.0 - s_alpha**2
        if c_alpha2 != 0.0:
            c_sigM2 = c_sig - 2.0*s_u0*s_u1/c_alpha2    # eq 18
        else:
            c_sigM2 = c_sig
        C = f/16.0 * c_alpha2*(4 + f*(4 - 3*c_alpha2))  # eq 10
        lambdaP = dL
        # dL => equation 11
        dL = L + (1 - C)*f*s_alpha*(sigma +
                                    C*s_sig*(c_sigM2 +
                                             C*c_sig*(-1.0 + 2*c_sigM2**2)))
        #
        if (cnt == max_iter):          # is it time to bail?
            return 0.0
        elif((math.fabs(dL - lambdaP) > 1.0e-12) and (cnt < max_iter)):
            cnt += 1
        else:
            break
    # ---- end of while ----
    uSq = c_alpha2 * ab_b
    A = 1 + uSq/16384.0 * (4096 + uSq*(-768 + uSq*(320 - 175*uSq)))  # eq 3
    B = uSq/1024.0 * (256 + uSq*(-128 + uSq*(74 - 47*uSq)))          # eq 4
    d_sigma = B*s_sig*(c_sigM2 +
                       (B/4.0)*(c_sig*(-1 + 2*c_sigM2**2) -
                        (B/6.0)*c_sigM2*(-3 + 4*s_sig**2)*(-3 + 4*c_sigM2**2)))
    # d_sigma => eq 6
    dist = b*A*(sigma - d_sigma)                                     # eq 19
    alpha1 = math.atan2(c_u1*s_dL, cs_01 - sc_01*c_dL)
    alpha2 = math.atan2(c_u0*s_dL, -sc_01 + cs_01*c_dL)
    # normalize to 0...360  degrees
    alpha1 = math.degrees(math.fmod((alpha1 + twoPI), twoPI))        # eq 20
    alpha2 = math.degrees(math.fmod((alpha2 + twoPI), twoPI))        # eq 21
    if verbose:
        return dist, alpha1, alpha2, cnt
    return dist, alpha1, alpha2, None


def _demo():
    """
    : -
    """
    pass
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    _demo()
