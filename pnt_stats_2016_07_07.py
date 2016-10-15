# coding: utf-8
"""
Script:  pnt_stats.py
Author:  Dan.Patterson@carleton.ca
Modified: 2016-07-07
Purpose:
:  return properties for xy point arrays
:Useage
:  xy a 2D array of x,y coordinates
:  f = 'pnt_10.txt'
:  np.savetxt(f, xy, fmt='%10i',delimiter=',')
:  np.loadtxt(f, delimiter=',')
:
:Notes
:  np.median  numpy.lib.function_base
:  np.mean    numpy.core.fromnumeric
:  np.min     ...
:  np.max     ...
:  np.ptp     ...
:  np.std     ...
:  np.var     ...
:  std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)
:  see _methods, all except median are delivered as methods as well
:  ie a.mean(axis=0) = np.mean(a, axis=0) etc
"""    
import numpy as np
import sys
import os

script = sys.argv[0]

def stats_xy(a, prn=True):
    """Data properties for 2D point/xy objects
    :Requires
    :--------
    :  The data should represent point objects in 2D.
    :  ie. a.shape: (n,2), a.ndim: 2 
    :Returns
    :-------
    :  General statistics, including extent values from which extent
    :  points can be formed.
    :Notes:
    :  It should be obvious that the 'result' list contains the methods
    :  to use if single values of a statistic are needed.
    :  - array construction below, querying follows
    :  - r[r['stat']=='mean']
    :      array([('mean', [300053.9, 5025068.3])], 
    :      dtype=[('Stat', '<U5'), ('XY', '<f8', (2,))])
    """
    if (a.ndim != 2):
        frmt ="{}: expects a 2D array... here are the docs\n\n{}" 
        print(frmt.format(stats_xy.__name__, stats_xy.__doc__))
        return None
    stat = "mean, median, min, max, ptp, std, var"
    names = [ s.strip() for s in stat.split(",") ]
    result = [np.mean(a, axis=0), np.median(a, axis=0),
              np.min(a, axis=0), np.max(a, axis=0), np.ptp(a, axis=0),
              np.std(a, axis=0, dtype=np.float64, ddof=0),
              np.var(a, axis=0, dtype=np.float64, ddof=0)
              ]
    r = list(zip(names, result))
    L, B = result[2]; R, T = result[3]
    frmt = "\nArray properties... N={}".format(len(a))
    frmt += "\n  L {}  B {}\n R  {}  T {}\n".format(L, B, R, T)
    frmt += "\n".join(['{!s:<8} {!s:<16}'.format(i,j) for i,j in r])
    if prn:
        print(frmt)
    else:
        dt = [('stat', 'U8'),("XY", "<f8", (2,))]
        r = np.asarray(r, dtype=dt)
        return r
    # ----
    
if __name__=="__main__":
    """using demo data"""
    path = sys.argv[0]
    f = os.path.dirname(path)+'/pnt_10.txt'
    a = np.loadtxt(f, delimiter=',')
    r = stats_xy(a, prn=True)
"""
sample data as an option to the file
    300078,   5025070
    300074,   5025099
    300086,   5025076
    300045,   5025089
    300015,   5025039
    300018,   5025074
    300062,   5025073
    300051,   5025023
    300061,   5025088
    300049,   5025052
"""
