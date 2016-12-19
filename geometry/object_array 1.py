# -*- coding: UTF-8 -*-
"""
:Script:  object_array.py
:Author:  Dan.Patterson@carleton.ca
:Modified: 2016-12-11
:Purpose: to demonstrate recursion... a script calls itself to perform a task
:  - alter the lines below and modify the list of lists to see the results
:
:Functions:
:---------
:  Geometry, such as polygons, can consist of single parts or multiparts.
:  In the case of polygons, there can be interior ring/holes as well. 
:  This can result in an unequal number of points per part making a uniform
:  numpy array impossible.  Conversion of the list of points to form an array
:  results in an array with an 'object' ('O') dtype.  This script attempts
:  to show ways of processing this data.
:
:  There are two variants of this, 'unpack' and 'parse_obj'.  The former goes
:  straight to an array of xy values with coordinates being flattened then
:  reshaped.  The parse_obj approach, retains the constituent subarrays 
:  should you want them for other purposes, such as removing donut holes
:  from polygons.
:
:  A final example of how to calculate area for a polygon with multiparts is
:  given using object arrays.
:
:Examples:
:--------  Polygon parts (see demo for more examples)
:  a0 = [[0, 0], [0, 8], [8, 8], [8, 0], [0, 0]]  # outer square
:  a1 = [[1, 1], [6, 1], [1, 6], [1, 1]]          # inner ring, left
:  a2 = [[2, 7], [7, 7], [7, 2], [2, 7]]          # inner ring, right
:  ---- form the array, unpack, reshape and derive the coordinates ----
:  a = [a0, a1, a2]
:  xy = unpack(a)  
:  xy = np.array(xy).reshape(len(xy)//2, 2)
:  x = xy[:,0]
:  y = xy[:, 1]
:  x => array([0, 0, 8, 8, 0, 1, 6, 1, 1, 2, 7, 7, 2])
:  y => array([0, 8, 8, 0, 0, 1, 1, 6, 1, 7, 7, 2, 7])
:  xy => array([[0, 0],
:               [0, 8],
:               [8, 8],
:               .......
:               [7, 7],
:               [7, 2],
:               [2, 7]])  
:   xy = np.vstack(parse_obj(a))  # ---- the same ----
:
:Notes:
:-----
: (1) type or instance properties... what an object is
:   - isinstance(obj, (list, tuple, np.ndarray))
:
: (2) attribute properties... what it has or what it can do
:   - pass the object to make sure... hasattr(object, parameter)
:   - e.g. ...             ---- object ----
:                  {}   []   ()  'a'   ndarray
:   '__getitem__'  X    X    X    X    X
:   '__iter__'     X    X    X    X    X
:   '__len__'      X    X    X    X    X
:   '__sizeof__'   X    X    X    X    X   (None has this as well)
:   'count'        -    X    X    X    -
:   'shape'        -    -    -    -    X
:   'size'         -    -    -    -    X
:References:
:----------
:  http://stackoverflow.com/questions/7200878/python-list-of-np-arrays
:       -to-array
:  http://stackoverflow.com/questions/1165647/how-to-determine-if-a- 
:       list-of-polygon-points-are-in-clockwise-order (signed area)
:  From the accepted answer...
:  Sum over the edges, (x2 − x1)(y2 + y1). If the result is positive the
:  curve is clockwise, if it's negative the curve is counter-clockwise.
:  (The result is twice the enclosed area, with a +/- convention.)
:---------------------------------------------------------------------:
"""
#---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent, indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=30, 
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

#---- functions ----

def unpack(iterable, param='__iter__'):
    """Unpack an iterable based on the param(eter) condition using recursion.
    :Notes:
    : ---- see main docs for more information and options ----
    : To produce an array from this, use the following after this is done.
    :   out = np.array(xy).reshape(len(xy)//2, 2)
    """ 
    xy = []
    for x in iterable:
       if hasattr(x, '__iter__'):
           xy.extend(unpack(x))
       else:
           xy.append(x)
    return xy
    

def parse_obj(obj_in, param="__len__", out=None, verbose=False):
    """Parse an multipart list, or array into its constituent parts using
    :  recursion.
    :Requires:
    :--------
    :  obj - list, tuple, ndarray of potentially nested components
    :  param - see main documentation
    :  out - the list to contain the arrays
    :  verbose - True to follow progress for testing.
    :Notes:
    :----- If the input array has a dtype of 'object' or 'O', then it has
    :      been constructed from nested lists of list of unequal size.  This
    :  attempts to break those down into constituent components and returns
    :  a list of arrays for further processing.
    :References:
    :----------
    : http://stackoverflow.com/questions/7200878/python-list-of-np-arrays
    :      -to-array
    """
    if hasattr(obj_in, param):
        if out is None:  out = []
        obj_in = np.asarray(obj_in).squeeze()
        for obj in obj_in:
            obj = np.asarray(obj).squeeze()
            if obj.dtype == 'O':
                ret = parse_obj(obj, out=out)
                if verbose:  print("Invalid {}\n{}".format(type(ret), ret))
                obj=ret
            else:
                out.append(obj)
                if verbose:  print("Valid {}\n{}".format(obj.dtype, obj))
        idx = np.array([len(i) for i in out])
    else:
       out = obj_in
       idx = None 
    return out, idx


def pnt_area(obj):
    """Calculate polygon area... see ein_geom for other options.
    """
    def cal_area(a):
        """calculation of parts"""
        a = np.asarray(a)
        x = a[:, 0]
        y = a[:, 1]
        return 0.5*(np.dot(x[1:], y[:-1]) - np.dot(y[1:], x[:-1]))
    #
    obj = np.asarray(obj)
    area = 0.0
    if (obj.dtype=='O') or (obj.ndim > 2):
        area = np.sum([cal_area(i) for i in obj])
    elif obj.ndim == 2:
        area = cal_area(obj)
    return area

def demo():
    """Demonstrates the functions"""
    a0 = [[0, 0], [0, 8], [8, 8], [8, 0], [0, 0]]  # outer square
    a1 = [[1, 1], [6, 1], [1, 6], [1, 1]]          # inner ring, left
    a2 = [[2, 7], [7, 2], [7, 7], [2, 7]]          # inner ring, right
    a = [a0, a1, a2]                               # outer, 2 inner
    b = [[10, 0], [10, 10], [20, 20], [20, 0], [10, 0]]  # outer square
    c = [[30, 0], [30, 10], [40, 0], [30, 0]]      # triangle
    d = [[a, b], [c, a]]                           # multipart with holes
    # ---- use the following multipart shape ----
    r0 = [[a0, a1, a2], b, c]                      # another example
    r1 = unpack(r0)
    r1 = np.array(r1).reshape(len(r1)//2, 2)
    r2, idx = parse_obj(r0, verbose=False)
    print("Input data...")
    en = list(enumerate(r0))
    for i in en:
        print("list ({})...\n{}".format(i[0], indent(str(i[1]), "    ")))
    en = list(enumerate(r2))
    print("\nArray coordinates (transformed... a.T)")
    for i in en:    # a.T 
        print(" part ({})...\n{}".format(i[0], indent(str(i[1].T), "   ")))
    print("\nCoordinates...\nx's: {}\ny's: {}".format(*np.vstack(r2).T))
    idx = np.cumsum(idx)
    r3 = np.split(r1, idx)[:-1]  # or split off he last idx
    print("\nArea calculation....")
    for i in range(len(r2)):
        print("Area for r2[{}] {:> 6.1f}".format(i, pnt_area(r2[i])))
    print("Total area = {}".format(pnt_area(r2)))
    return r0, r1, r2, idx


if __name__ == '__main__':
    """recursion demo  
    """
    r0, r1, r2, idx = demo()
    
