# -*- coding: UTF-8 -*-
"""
:Script:   strides_demo.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-11-25
:Purpose:
:-------   strides in arrays
:Functions:  
:---------
: mine...
: -  _check(a, r_c)  -  check array and return 
: -  stride(a, r_c=(3, 3))
: as_strided, modified from numpy.lib.stride_tricks import
: -  _maybe_view_as_subclass(original_array, new_array)
: -  view_ast(x, shape=None, strides=None, subok=False, writeable=True)
: -  _demo()
:Notes:
:
:References:
:
:---------------------------------------------------------------------:
"""
#---- imports, formats, constants ----

import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
from textwrap import dedent, indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ["_check",
           "stride",
           "_maybe_view_as_subclass",
           "view_ast",
           "_demo"]

#---- functions ----


def _check(a, r_c, subok=False):
    """Performs the array checks necessary for stride and block.
    : a   - Array or list.
    : r_c - tuple/list/array of rows x cols.
    : subok - from numpy 1.12 added, keep for now
    :Returns:
    :------
    :Attempts will be made to ...
    :  produce a shape at least (1*c).  For a scalar, the
    :  minimum shape will be (1*r) for 1D array or (1*c) for 2D
    :  array if r<c.  Be aware
    """
    if isinstance(r_c, (int, float)):
        r_c = (1, int(r_c))
    r, c = r_c
    if a.ndim == 1:
        a = np.atleast_2d(a)
    r, c = r_c = ( min(r, a.shape[0]), min(c, a.shape[1]) )
    a = np.array(a, copy=False, subok=subok)
    return a, r, c, tuple(r_c)

    
def stride(a, r_c=(3, 3)):
    """Provide a 2D sliding/moving view of an array.  
    :  There is no edge correction for outputs.
    :
    :Requires
    :--------
    : _check(a, r_c) ... Runs the checks on the inputs.
    : a - array or list, usually a 2D array.  Assumes rows is >=1,
    :     it is corrected as is the number of columns.
    : r_c - tuple/list/array of rows x cols.  Attempts  to 
    :     produce a shape at least (1*c).  For a scalar, the
    :     minimum shape will be (1*r) for 1D array or 2D
    :     array if r<c.  Be aware
    """
    a, r, c, r_c = _check(a, r_c)
    shape = (a.shape[0] - r + 1, a.shape[1] - c + 1) + r_c
    strides = a.strides * 2
    a_s = (as_strided(a, shape=shape, strides=strides)).squeeze()
    return a_s


def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


class DummyArray(object):
    """Dummy object that just exists to hang __array_interface__ dictionaries
    :  and possibly keep alive a reference to a base array. From numpy 1.12
    """
    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base


def view_ast(x, shape=None, strides=None, subok=False, writeable=True):
    """ View an array as a strided array.
    :Modified from numpy 1.12
    :Returns:
    :-------
    :  A view or copy of an array with a different stride
    :References:
    :----------
    :- https://github.com/numpy/numpy/blob/master/
    :         numpy/lib/stride_tricks.py
    :  - from numpy.lib.stride_tricks import as_strided
    :
    :-  https://geonet.esri.com/blogs/dan_patterson/2016/07/07/
    :          slidingmoving-window-operations-in-rasters-and-arrays
    :-  http://stackoverflow.com/questions/40773275/sliding-standard-
    :         deviation-on-a-1d-numpy-array
    :Notes:
    :-----
    :  ain = dict(a.__array_interface__)
    :  ain.items()  => (key, value) => ain.keys(), ain.values()
    :  dict_items([('version', 3), ('descr', [('', '<i8')]),
    :              ('strides', None), ('data', (6176702208, False)),
    :              ('shape', (10,)), ('typestr', '<i8')])
    :
    """
    x = np.array(x, copy=False, subok=subok)
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    array = np.asarray(DummyArray(interface, base=x))
    if array.dtype.fields is None and x.dtype.fields is not None:
        # This should only happen if x.dtype is [('', 'Vx')]
        array.dtype = x.dtype
    view = _maybe_view_as_subclass(x, array)
    if view.flags.writeable and not writeable:
        view.flags.writeable = False
    return view, list(interface.items())


def _demo():
    """demo using the original code of as_strided
    :  view_ast = as_strided
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    :input array
    {}
    :
    :array interface dictionary items...
    {}
    :
    :Sample calculation using window of {}
    :result...
    {}
    :
    :strided array
    {}
    :
    :output array interface dictionary items
    {}
    :
    :------------------------------------------------------------------
    """
    p = "    "
    a = np.random.randint(0,5, size=10)
    a_items = list(dict(a.__array_interface__).items())
    a_txt = "\n".join(["    {!s:}".format(i) for i in a_items])
    W = 3 # Window size
    nrows = a.size - W + 1
    n = a.strides[0]
    a2D, int_items = view_ast(a, shape=(nrows, W), strides=(n, n))
    o_txt = "\n".join(["    {!s:}".format(i) for i in int_items])
    out = np.sum(a2D, axis=1)
    args = ["Sliding windows... based on as_strided...",
            indent(str(a), p),
            a_txt, W,
            indent(str(out), p),
            indent(str(a2D), p), o_txt]
    print(dedent(frmt).format(*args))
    #return a, a2D, int_items

    
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    #a, a2D, int_items = _demo()
    _demo()
#Z = np.random.randint(0,5,(10,10))
"""
a = np.arange(1, 37).reshape(6, 6)
n = 3
m = 3
i = 1 + (a.shape[0]-n)  # 3
j = 1 + (a.shape[1]-m)  # 3
v, ain = view_ast(a, shape=(i, j, n, m), strides=a.strides + a.strides) # n,n
template = np.ones((3,3),dtype='int')
result = np.einsum('ijkl,kl->ij', v, template)
result2 = np.sum(v, axis=(2,3))
# result = result2
"""
