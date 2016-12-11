# -*- coding: UTF-8 -*-
"""
:Script:   arr_row.py
:
:Author:   Dan.Patterson@carleton.ca
:
:Modified: 2016-08-26
:
:Purpose:
:  The _f function is used to provide a side-by-side view of 2,3, and 4D
:  arrays.  Specifically, 3D and 4D arrays are useful and for testing 
:  purposes, seeing the dimensions in a different view can facilitate
:  understanding.  For the best effect, the array shapes should be carefully
:  considered. Some guidelines follow.  The middle 'r' part of the shape is
:  not as affected as the combination of the 'd' anc 'c' parts.  The array is
:  trimmed beyond the 'wdth' parameter in _f.
:
:  Sample the 3D array shape so that the format (d, r, c)
:  is within the 20-21 range for d*c ... for example:
:        integers          floats
:        2, r, 10  = 20    2, r, 8 = 16
:        3, r,  7  = 21    3, 4, 5 = 15
:        4, r,  5  = 20    4, r, 4 = 16
:        5, r,  4  = 20    5, r, 3 = 15
:
:   _f(a)  example for a =  np.arange(3*4*7).reshape(3, 4, 7)
:  ---------------------------------------------------
:  : Array... shape (3, 4, 5), ndim 3
:  :
:    0  1  2  3  4    20 21 22 23 24    40 41 42 43 44   
:    5  6  7  8  9    25 26 27 28 29    45 46 47 48 49   
:   10 11 12 13 14    30 31 32 33 34    50 51 52 53 54   
:   15 16 17 18 19    35 36 37 38 39    55 56 57 58 59   
:  : sub (0 )        : sub (1 )        : sub (2 )
:
:  The middle part of the shape should also be reasonable should you want
:  to print the results:
:
:  How it works
:
:  a[...,0,:].flatten()
:  array([ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44])
:
:  a[...,0,(0, 1, -2, -1)].flatten()
:  array([ 0,  1,  3,  3, 20, 21, 23, 23, 40, 41, 43, 43])
:
:Functions:  help(<function name>) for help
:---------
: _f - see this function for more documentation (ie help(_f))
:    - _check, _frmt, _prn  private functions
: _bigdemo, _smalldemo -  Sample functions ...
:
:Notes:
:-----
:
:References:
:----------
:
:----------
"""
#---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
script = sys.argv[0]

#---- functions ----

def _f(a, wdth=80, title="Array"):
    """Format number arrays by row.
    :Requires:
    :--------
    : a - Array of int or float dtypes, 2, 3 and 4D arrays only.
    :   The array should be a small one, it is designed for demonstration
    :   purposes.  See the Notes section for suggested array sizes with
    :   respect to (d, r, c).
    : wdth - Default width for onscreen and printing, output beyond this
    :   length will be truncated with a warning.  Reshape to overcome.
    : title - The default title, change to provide more information.
    :
    :Returns:
    :-------
    : Print the array with the 1st dimension flattened-like by row
    :
    :Notes:
    :-----
    : w_frmt - width formatter
    : m_frmt - max number formatter to get max. number of characters
    :
    : See script docs for more information.
    :
    :-------
    """
    a = np.asarray(a)
    # ---- private functions ----

    def _check(a):
        """ check dtype and max value for formatting information"""
        if a.ndim < 3:
            shp = (1,) + a.shape
            a = np.reshape(a, shp)
        d, r, c = a.shape[-3:]
        dt_k = a.dtype.kind
        a_max = np.max(a)
        return d, r, c, dt_k, a_max

    def _frmt(dt_k, a_max):
        """format size of column based on max. value and dtype"""
        w_, m_ = [[':{}.0f', '{:0.0f}'],
                  [':{}.1f', '{:0.1f}']][dt_k == 'f']  # is float?
        m_frmt = len(m_.format(a_max)) + 1
        w_frmt = w_.format(m_frmt)
        return w_, m_, m_frmt, w_frmt

    def _prn(hdr0, rows, row_frmt, wdth=wdth, is_first=False):
        """print the subset"""
        if is_first:
            len_s = len(row_frmt.format(*rows[0]).rstrip())
            len_frst = min(len_s, wdth)
            if len_s > wdth:
                print("\nOutput being trimmed to {} characters".format(wdth))
            print("\n" + "-"*len_frst + hdr0)
        for vals in rows:
            v = row_frmt.format(*vals)
            print(v[:wdth])
    #
    # now you can begin
    #
    d, r, c, dt_k, a_max = _check(a)
    w_, m_, m_frmt, w_frmt = _frmt(dt_k, a_max)
    #
    approx =  int(100/((len(str(a.max())) + 2) * 3))
    max_c = min(c, approx)  # 5 hard-coded
    if c >= approx:
        cols = 5
        max_c = c
        full = False
    else:
        cols = c
        max_c = min(c, approx)
        full = True
    s = (m_frmt + 1)*cols + 1
    hdr = ((': sub ({:<1.0f})' + " "*s)[:s+4])*d
    row_frmt = (('{'+ w_frmt +'}')*cols + '   ')*d
    row_frmt = row_frmt.replace(', ', ' ')
    # format line with dashes and string
    hdr0 = "\n: {}... shape {}, ndim {}\n:".format(title, a.shape, a.ndim)
    if (a.ndim == 3):
        d, r, c = a.shape
        if full:  # or hard-coded to 5
            rows = [a[...,i,:].flatten() for i in range(r)]
        else:
            tail = " "*m_frmt
            rows = [a[...,i,(0,1,-2,-1)].flatten() for i in range(r)]
            row_frmt = (('{'+ w_frmt +'}')*2 + ' ..' +\
                        ('{'+ w_frmt +'}')*2 + tail)*d
        #
        _prn(hdr0, rows, row_frmt, wdth, is_first=True)
        #
        h = [": sub ({:<1.0f} )".format(i) for i in range(d)]
        L = m_frmt*cols + 3
        hf = ("{!s:<" + str(L) + "}")*d
        print("".join([hf.format(*h)])[:wdth])
    else:
        d4, d, r, c = a.shape
        cnt = True
        for d3 in range(d4):
            a_s = a[d3]
            rows = [a_s[...,i,:].flatten() for i in range(r)]
            #
            _prn(hdr0, rows, row_frmt, wdth, is_first=cnt)
            #
            hdr0 = ":--- array[{},...] => ({}, {}, {})\n".format(d3, d, r, c)
            cnt = False
            print(hdr0.format(*(np.arange(d))))
    return None


def _bigdemo():
    """
    :Requires:
    :--------
    :
    :Returns:
    :-------
    :
    """
    cnt = 0
    fac = 1  # change to 10, 100 etc or 1.0, 10.0, 100. for floats
    shp = [[4, 4, 5], [4, 4, 7], [5, 4, 5], [3., 4, 10]]
    for d, r, c in shp:
        a = np.arange(d*r*c).reshape(d, r, c) * fac
        t = "Array...{}".format(cnt)
        _f(a, wdth=100, title=t)
        cnt += 1
    return a


def _smalldemo():
    """ small samples """
    d, r, c = (3, 4, 7)
    fac = 1
    a = np.arange(d*r*c).reshape(d, r, c) * fac
    b = a*10
    c = np.arange(2*4*3*4).reshape(2, 4, 3, 4)*10
    _f(a, wdth=100, title="Array a")
    _f(b, wdth=100, title="Array b")
    _f(c, wdth=100, title="Array c, 4D")
    return a


#-------------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    a = _bigdemo()
    #a = _smalldemo()

