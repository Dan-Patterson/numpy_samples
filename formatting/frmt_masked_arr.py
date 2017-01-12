# -*- coding: UTF-8 -*-
"""
:Script:   frmt_masked_arr.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-01-08
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
:  to retrieve the mask values if they aren't known, then try
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
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

#---- function ----
def frmt_ma(a, prn=True, prefix="  ."):
    """Format a masked array to preserve columns widths and style.
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
    def _fix(v, tmp, prefix):
        """ sub array adjust"""
        r = [['[['," "], ['[', ""], [']', ""], [']]', ""]]
        for i in r:
            tmp = tmp.replace(i[0], i[1])
        tmp0 = [i.strip().split(' ') for i in tmp.split('\n')]        
        N = len(tmp0[0])
        out = [""]
        for i in range(len(tmp0)):
            out.append((ft*N).format(*tmp0[i]))
        jn = "\n" + prefix
        v += jn.join([i for i in out])       
        v += '\n'
        return v
    #
    dim = a.ndim
    shp = a.shape
    a_max = len(str(np.ma.max(a)))
    ft = '{:>' + str(a_max + 1) + '}'
    v = "\n:Masked array... ndim: {}\n".format(dim)
    if dim == 2:
        v += "\n:.. a[:{}, :{}] ...".format(*shp)
        v = _fix(v, str(a), prefix)
    elif dim == 3:
        for d0 in range(shp[0]): # dimension blocks
            v += "\n:.. a[{}, :{}, :{}] ...".format(d0, *a[d0].shape)
            v = _fix(v, str(a[d0]), prefix)
    if prn:
        print(v)
    return v


#---- demos ----
def _data():
    """Produce a simple masked array.  Change the values to suit
    """
    a = np.array([[ 100, 1, 2, -99,   99], [ 5, 6, 7, -99, 9],
       [-99,  11, -99,  13, -99], [ 15,  16,  17,  18,  19],
       [-99, -99,  22,  23,  24], [ 25,  26,  27,  28, -99],
       [ 30,  31,  32,  33, -99], [ 35, -99,  37,  38,  39]])
    m = np.where(a == -99, 1, 0 )
    mask_val = -99
    a = np.ma.array(a, mask=m, fill_value=mask_val)
    return a


def _ma_demo():
    """Produce a simple masked array and format it using frmt_ma
    :  Change the values to suit
    """
    np.ma.masked_print_option.set_display('-')
    a = np.array([[ 100, 1, 2, -99,   99], [ 5, 6, 7, -99, 9],
       [-99,  11, -99,  13, -99], [ 15,  16,  17,  18,  19],
       [-99, -99,  22,  23,  24], [ 25,  26,  27,  28, -99],
       [ 30.,  31,  32,  33, -99], [ 35, -99,  37,  38,  39]], dtype='<f8')
    m = np.where(a == -99, 1, 0 )
    mask_val = -99
    a = np.ma.array(a, mask=m, fill_value=mask_val)
    # ---- test output ----
    print("Sample run of frmt_ma...")
    frmt_ma(a, prn=True)
    print("\nArray reshaped two 3D")
    b = a.reshape(2,4,5)
    frmt_ma(b, prn=True)
    return a, b


#----------------------
if __name__ == "__main__":
    """   """
    #print("Script... {}".format(script))
    a, b = _ma_demo()

