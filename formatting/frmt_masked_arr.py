# -*- coding: UTF-8 -*-
"""
:Script:   frmt_masked_arr.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-07-25
:Purpose:
:  This sample produces a n*m shaped array which you could pass parameters
:  to, but it is simpler to edit them here. It then produces a random
:  choice set of locations and sets them to a value which is masked and a
:  true masked array is produced.
:  The location of the selection and their original values is also returned.
:
:Sample
:  shp = (5,4)
:  a = np.arange(np.prod(shp)).reshape(shp)
:  m = np.random.randint(0, np.prod(shp), 5)
:  array([ 7, 16,  8,  0, 18])
:  ix = np.in1d(a.ravel(), m).reshape(a.shape)
:  array([[1, 0, 0, 0],
:         [0, 0, 0, 1],
:         [1, 0, 0, 0],
:         [0, 0, 0, 0],
:         [1, 0, 1, 0]], dtype=bool)
:  a_m = np.ma.array(a, mask=ix, fill_value=-1)
:
:Retrieving mask values:
:  if you need to retrieve the mask values and they aren't know, then you can try
:
:  msk = np.unique(c._data.flatten()*c._mask.flatten())
:
:Example:
:-------
:  shp = (5,4)
:  a = np.arange(np.prod(shp)).reshape(shp)
:  m = np.random.randint(0, np.prod(shp), 5)
:    = array([ 7,  5,  1,  5, 15])
:
:  the array                     the mask
:  array([[ 0,  -,  2,  3],       [[0 1 0 0],
:         [ 4,  -,  6,  -],        [0 1 0 1],
:         [ 8,  9, 10, 11],        [0 0 0 0],
:         [12, 13, 14,  -],        [0 0 0 1],
:         [16, 17, 18, 19]])       [0 0 0 0]], fill_value = -1
:
:  ix = np.in1d(a.ravel(), m).reshape(a.shape)
:  array([[0, 1, 0, 0],
:         [0, 1, 0, 1],
:         [0, 0, 0, 0],
:         [0, 0, 0, 1],
:         [0, 0, 0, 0]], dtype=bool)
:
: - Produce the masked array as shown above.  Retrieve the mask if unknown.
:
:  a_m = np.ma.array(a, mask=ix, fill_value=-1)
:  
:  msk = np.unique(a_m._data.flatten()*a_m._mask.flatten())
:      = array([ 0,  1,  5,  7, 15])
:
"""
#---- imports, formats, constants ----
import sys
import numpy as np
from textwrap import dedent
import arr_tools as arr

ft = {'bool':lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)

script = sys.argv[0]

#---- function ----
def frmt_ma(a, prn=True):
    """
    :Requires
    :--------
    : Input a masked array.  Get a string representation of the array.
    : Determine the maximum value and format each column using that value.
    :
    :Returns
    :-------
    : Returns a print version of a masked array formatted with masked
    : values and appropriate spacing.
    : b = a.reshape(2,4,5) for 3d
    """
    def _fix(v, tmp):
        """ sub array adjust"""
        r = [['[['," "], ['[', ""], [']', ""], [']]', ""]]
        for i in r:
            tmp = tmp.replace(i[0], i[1])
        tmp0 = [i.strip().split(' ') for i in tmp.split('\n')]        
        N = len(tmp0[0])
        out = []
        for i in range(len(tmp0)):
            out.append((ft*N).format(*tmp0[i]))
        v += "\n".join([i for i in out])       
        v += '\n'
        return v
    #
    frmt = """
    :--------------------
    :Masked array........
    :  ndim: {} size: {}
    :  shape: {}
    """
    shp = a.shape
    dim = a.ndim
    a_max = len(str(np.ma.max(a)))
    ft = '{:>' + str(a_max + 1) + '}'
    f = dedent(frmt).format(a.ndim, a.size, a.shape)
    v = f #+ "\n"
    if dim == 2:
        tmp = str(a)
        v += ":\n:... a[:{}, :{}] ...\n".format(*a.shape)
        v = _fix(v, tmp)
    elif dim == 3:
        case = shp[0]
        for cs in range(case):
            tmp = str(a[cs])  
            v += ":\n:... a[{}, :{}, :{}] ...\n".format(cs, *a[cs].shape)
            v = _fix(v, tmp)
    if prn:
        print(v)
    return v
#
#---- demos ----
def _demo():
    """
    :Returns
    :-------
    : A sample masked array for frmt_ma. Change the values to suit
    """
    np.ma.masked_print_option.set_display('-')
    a = np.array([[ 100, 1, 2, -99,   99], [ 5, 6, 7, -99, 9],
       [-99,  11, -99,  13, -99], [ 15,  16,  17,  18,  19],
       [-99, -99,  22,  23,  24], [ 25,  26,  27,  28, -99],
       [ 30,  31,  32,  33, -99], [ 35, -99,  37,  38,  39]])
    m = np.where(a == -99, 1, 0 )
    mask_val = -99
    a = np.ma.array(a, mask=m, fill_value=mask_val)
    return a

def _demo2():
    """ another sample"""
    shp = (5,4)
    a = np.arange(np.prod(shp)).reshape(shp)
    m = np.random.randint(0, np.prod(shp), 5)
    ix = np.in1d(a.ravel(), m).reshape(a.shape)
    a_m = np.ma.array(a, mask=ix, fill_value=-1)
    a = frmt_ma(a_m, prn=True)
    return a_m

#----------------------
if __name__ == "__main__":
    """   """
    #print("Script... {}".format(script))
    a = _demo()
    a_ma = frmt_ma(a, prn=True)
    b = a.reshape(2,4,5)
    b_frmt = frmt_ma(b, prn=True)
    c = _demo2()
