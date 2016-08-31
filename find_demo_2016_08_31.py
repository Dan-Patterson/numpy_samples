"""
:Script:   find_demo.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-08-31
:Purpose:  Essentially a demo wrapper function around np.where and
:  np.in1d, for demonstrations purposes.  It also demonstrates the
:  steps one needs to take to ensure queries are formulated correctly.
:
:Notes:   see dup_seq for data sources
:  np.in1d  added in version 1.8 use,
:  np.lib.arrayset_ops.in1d for older versions
:
: .... Other queries involving 'where' ....
: .... Floats ....
: a = np.arange(25).reshape(5,5)           # used in subsequent examples
: array([[ 0,  1,  2,  3,  4],
:        [ 5,  6,  7,  8,  9],
:        [10, 11, 12, 13, 14],
:        [15, 16, 17, 18, 19],
:        [20, 21, 22, 23, 24]])
:
: np.where( (a<5) | (a>20), np.NaN, a )    # the 'or' case
: array([[ nan,  nan,  nan,  nan,  nan],
:        [  5.,   6.,   7.,   8.,   9.],
:        [ 10.,  11.,  12.,  13.,  14.],
:        [ 15.,  16.,  17.,  18.,  19.],
:        [ 20.,  nan,  nan,  nan,  nan]])
:
: .... Integers ....
:  r = a.ravel() 
:  look_for = np.array([2, 5, 9])     # conditions to look for
:  zz = [ -99 if val in look_for      # using a list comp
              else val 
              for val in r  ]
:  zz = np.array(zz).reshape(a.shape) # some reshaping and we are done
:  array([[  0,   1, -99,   3,   4],
:         [-99,   6,   7,   8, -99],
:         [ 10,  11,  12,  13,  14],
:         [ 15,  16,  17,  18,  19],
:         [ 20,  21,  22,  23,  24]])
:
: .... in1d examples ....
:  look_for = [2, 10, 15, 16, 20]
:  u = np.in1d(a[:,0], look_for)  # look only in the first 2 columns
:  v = np.in1d(a[:,1], look_for)
:  rs = np.sum((u,v),axis=0)
:
:  result = a[np.where(rs==2)]    # returns the whole row where 2 found
:  array([[15, 16, 17, 18, 19]])
:
:  result = a[np.where(rs>=1)]    # returns where 1 or more found
:  array([[10, 11, 12, 13, 14],
:         [15, 16, 17, 18, 19],
:         [20, 21, 22, 23, 24]])
:
: .... Find all the row/columns that our criteria are found ....
:  look_for = [2, 10, 15, 16, 20]
:  r = a.ravel()            
:  w = np.in1d(r, look_for)
:  w = w.astype('int').reshape(a.shape)
:  array([[0, 0, 1, 0, 0],
:         [0, 0, 0, 0, 0],
:         [1, 0, 0, 0, 0],
:         [1, 1, 0, 0, 0],
:         [1, 0, 0, 0, 0]])
:  ix = np.where(w)
:  (array([0, 2, 3, 3, 4]), array([2, 0, 0, 1, 0]))
:  found = np.vstack(ix).T         # yields the row/column pairs
:  array([[0, 2],
:         [2, 0],
:         [3, 0],
:         [3, 1],
:         [4, 0]])
:
:-----------------
"""
import numpy as np
from textwrap import dedent

ft={'bool': lambda x: repr(x.astype('int32')),
    'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=4, linewidth=80, precision=2,
                    suppress=True, threshold=5, 
                    formatter=ft) #formatter={'float': '{: 0.3f}'.format})
__all__ = ['find']
__private__ = ['_in_1D', '_in_N2', '_in_2D', '_demo']

# ---- find functions ----------------------------------------------
def _in_1D(a, look_for=None, switch=False, verbose=False):
    """See find_ docs""" 
    a = np.asarray(a)  
    result = np.in1d(a, look_for, invert=switch)
    found = a[result]
    return result, found


def _in_N2(a, look_for=None, switch=False, verbose=False):
    """See find_ docs """
    err = "Query N2 arrays with a pair or list of pairs"
    a = np.asarray(a)
    if (look_for.ndim == 0) or (len(look_for) < 2):
        print(err)
        return None, None
    elif (look_for.ndim == 1) and (len(look_for) != 2):
        print(err)
        return None, None
    else:
        look_for = np.atleast_2d(look_for)
    result = []
    for i in look_for:
        cond = np.where(a == i, 1, 0)
        rs = np.sum(cond, axis=1)
        temp = a[np.where(rs == 2)]
        result.extend(temp.tolist())
    found = np.asarray(np.vstack(result))
    return result, found

def _in_ND(a, look_for=None, switch=False, verbose=False):
    """See find_ docs """ 
    a = np.asarray(a)
    ix = np.in1d(a.ravel(), look_for, invert=switch).reshape(a.shape)
    result = np.where(ix)
    found = np.vstack(result).T  # or = np.array(list(zip(*np.where(ix))))
    return result, found

def find(a, look_for=None, switch=False, verbose=False):
    """Find entries in 1D and 2D arrays.  2D arrays can be N2 or NM shaped.
    :Requires:
    :--------
    : a        - an array 
    : look_for - a value, list or array of values to query the array.
    :    If a single value, or a singleton in a list is passed to a 2D 
    :    array, a warning will be printed.
    : switch   - perform a reverse on the selection
    : verbose  - True, to print the results
    :
    :Calls to:
    :--------
    : _in_1D 
    :   - 1D arrays queries, use switch=True to invert selection
    : _in_N2
    :   - 2D arrays with shape (N, 2) using (N, 2) queries 
    :     look_for = [[1, 2], [3, 4]]   Use in_2D if you want to query
    :     with singletons or values in lists.
    : _in_ND 
    :   - 2D arrays of any shape using single values or lists of
    :     queries
    :
    :Returns:
    :-------
    :    depends on the call.
    :
    :-----------------------------
    """
    frmt ="""
    :-----------------------------
    : Test array...
    : - ndim {}  shape {}
    {!r:}
    : - Look for ...: {!r:}
    : - Result: 
    {!r:}
    : - Found:
    {!r:}
    :-----------------------------
    """
    a = np.asarray(a)
    shp = a.shape
    dim = a.ndim
    found = ""   # placeholder
    if (look_for is None): 
        print("No conditions to check")
        return a
    look_for = np.asarray(look_for)
    # ---- checks done ----
    if dim < 2:  # call _find_1D
        result, found = _in_1D(a, look_for, switch, verbose)
    elif (dim == 2) and (shp[-1] == 2) and (look_for.ndim == 2):
        result = _in_N2(a, look_for, switch, verbose)
    elif (dim >= 2):
        result, found = _in_ND(a, look_for, switch, verbose)
    if verbose:
        args = [dim, shp, a, look_for, result, found]
        print(dedent(frmt).format(*args))
    return result, found

# ---- demo ----    
def _demo():
    """run with demo data"""
    #
    # .... 1D cases ....
    a0 = [0, 0, 1, 8, 2, 2, 4, 4, 4, 7,]
    look_for0 = [2, 4]
    result0 = find(a0, look_for0, switch=False, verbose=True)
    #
    # .... N2 cases ....
    X = [0., 0, 1, 8, 2, 2, 4, 4, 4, 7]  # split X from above
    Y = [0., 1, 8, 2, 2, 4, 4, 4, 7, 7]  # shifted left by 1
    look_for1 = [2, 4, 8]
    look_for2 = [[2, 4], [4, 4]]
    a1 = np.array(list(zip(X, Y)),dtype='float64')  # or np.vstack((X,Y)).T
    result1 = find(a1, look_for1, switch=False, verbose=True)
    result2 = find(a1, look_for2, switch=False, verbose=True)
    #
    # .... 2D cases ....
    a2 = np.arange(25).reshape(5, 5)
    look_for3 = [7, 8, 10, 13]
    result3 = find(a2, look_for3, switch=False, verbose=True)
    #
    # .... 3D case ....
    tmp = np.arange(4*5).reshape(4, 5)
    a3 = np.vstack((tmp, tmp, tmp)).reshape(3, 4, 5)
    look_for4 = [6, 14, 2, 17]
    result4 = find(a3, look_for4, switch=False, verbose=True)
#    result4 =""
    #z = [ i.tolist() for i in result4[0][0]]
    #zz= [i.tolist() for i in result4[0][1]]
    return a0, result0, a1, result1, result2, a2, result3, a3, result4

arr = np.array([[17,  5, 19,  9], [18, 13,  3,  7], [ 8,  1,  4,  2],
                [ 8,  9,  7, 19], [ 6, 11,  8,  5], [11, 16, 13, 18],
                [ 0,  1,  2,  9], [ 1,  7,  4,  6]])        


if __name__ == "__main__":
    """ Run the demo"""
    a0, result0, a1, result1, result2, a2, result3, a3, result4 = _demo()

