# -*- coding: UTF-8 -*-
"""
:Script:   ein_area.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-11-05
:
:Purpose: To obtain polygon area for multiple polygons, either with rings or
:   multipart in nature.
:
:Functions:  help(<function name>) for help
:---------
: _demo  -  This function ...
:
:Notes:
:-----
: Alternates to ein_area
: (1)  area1 = 0.5*np.abs(np.dot(x[1:], y[:-1]) - np.dot(y[1:], x[:-1]))
: (2)  area2 = 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
:
:References:  numpy solutions
:----------
:  - http://stackoverflow.com/questions/24467972/
:    calculate-area-of-polygon-given-x-y-coordinates/24468019#24468019
:  - https://en.m.wikipedia.org/wiki/Shoelace_formula
:  2D, 3D broadcasting issue
:  - http://stackoverflow.com/questions/35165149/
:    summing-over-ellipsis-broadcast-dimension-in-numpy-einsum
:  - https://github.com/uniomni/oles-tutorials/blob/master/scipy2012/
:           geoprocessing_tutorial/exercises/exercise4.py
:
:Wikipedia    (3,4), (5,11), (12,8), (9,5), and (5,6). area = 30
:  centroid, area
:------------------------------------------------------------------------------------
""" 
import sys
import numpy as np
import inspect
import arr_tools as art

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)

script = sys.argv[0]

#---- functions ----

def ein_area(a, b=None):
    """Area calculation, using einsum.
    :  Some may consider this overkill, but consider a huge list of polygons,
    :  many multipart, many with holes and even multiple version therein.
    :Requires:
    :--------
    :  a - either a 2D+ array of coordinates or arrays of x, y values
    :  b - if a < 2D, then the y values need to be supplied
    :  Outer rings are ordered clockwise, inner holes are counter-clockwise
    :Notes:
    :-----
    :  x => array([ 0.000,  0.000,  10.000,  10.000,  0.000])  .... OR ....
    :  t = x.reshape((1,) + x.shape)
    :      array([[ 0.000,  0.000,  10.000,  10.000,  0.000]]) .... OR ....
    :  u = np.atleast_2d(x)
    :      array([[ 0.000,  0.000,  10.000,  10.000,  0.000]]) .... OR ....
    :  v = x[None, :]
    :      array([[ 0.000,  0.000,  10.000,  10.000,  0.000]])
    :-----------------------------------------------------------------------
    """
    a = np.array(a)  
    if b is None:
        xs = a[..., 0]
        ys = a[..., 1]
    else:
        xs, ys = a, b
    x0 = np.atleast_2d(xs[..., 1:])
    y0 = np.atleast_2d(ys[..., :-1])
    x1 = np.atleast_2d(xs[..., :-1])
    y1 = np.atleast_2d(ys[..., 1:])
    e0 = np.einsum('...ij,...ij->...i', x0, y0)
    e1 = np.einsum('...ij,...ij->...i', x1, y1)
    area = abs(np.sum((e0 - e1)*0.5))
    return area


def _demo(dt='float64'):
    """Testing for separate polygons, those with holes and separate multipart
    : Example: 
    :  outer   x's: -10 - 10 = 20
    :          y's: -10 - 10 = 20 ... therefore area = 400
    :  inner   x's:  -5 -  5 = 10
    :          y's:  -5 -  5 = 10 ... therefore area = 100  ... final = 300    
    :
    """
    # outer (a,b,c)  inner (d,e,f)  holes( g)  multipart (h)
    a = np.array([[0, 0], [0, 10], [10,  10], [ 10, 0], [0, 0]], dtype=dt)
    b = np.array([[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]], dtype=dt)
    c = -np.array(a)
    d = np.array([[2, 2], [8, 2], [ 8, 8], [ 2, 8], [2, 2]], dtype=dt)
    e = np.array([[3, 3], [7, 3], [ 7, 7], [ 3, 7], [3, 3]], dtype=dt)
    f = np.array([[4, 4], [6, 4], [ 6, 6], [ 4, 6], [4, 4]], dtype=dt)
    g = np.array([[a, d], [b, f]])
    h = np.array([a, c])
    xy = [a, b, c, d, e, f, g, h]
    cnt = 1
    names = ["a_out", "b_out", "c_out",
             "d_in", "e_in", "f_in",
             "g_stacked holes [a,b], [d,f]", "h_separate_parts"]
    print("\nArea using... ein_area ...\n{}".format(_demo.__doc__))
    for i in xy:
        a = np.array(i, dtype=dt)
        args = [cnt, names[cnt-1], a.shape]
        frmt = "\n({}) Array\n {}... shape: {}"
        print(frmt.format(*args))
        print("\n{!r:}\narea {}\n{}".format(a, ein_area(a), "-"*60))
        cnt += 1
    return xy


#----------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    a, b, c, d, e, f, g, h = _demo(dt='float64')
