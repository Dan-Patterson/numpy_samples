# coding: utf-8
"""
Script:  reclass_array_demo.py
Author:  Dan.Patterson@carleton.ca
Modified: 2016-03-11
Purpose: To demonstrate how to perform basic array reclassification 
  with or with nodata values
References:
    RasterToNumPyArray and NumPyArrayToRaster can be used to and from
    raster formats of different types.
    

"""
import numpy as np
from textwrap import dedent

def arr_reclass(a, bins=[], new_bins=[], mask=False, mask_val=None):
    """a    - integer or floating point array to be reclassed using
       bins - sequential list/array of the lower limits of each class
          include one value higher to cover the upper range.
       mask - whether the raster contains nodata values or values to
          be masked with mask_val
       array dimensions will be squeezed 
    """ 
    a_rc = np.zeros_like(a)
    if (len(bins) < 2): # or (len(new_bins <2)):
        print("Bins = {} new = {} won't work".format(bins,new_bins))
        return a
    if len(new_bins) < 2:
        new_bins = np.arange(1,len(bins)+2)   
    new_classes = zip(bins[:-1],bins[1:],new_bins)
    for rc in new_classes:
        q1 = (a >= rc[0])
        q2 = (a < rc[1])
        z = np.where(q1 & q2, rc[2],0)
        a_rc = a_rc + z
    return a_rc

def demo():
    """run various demos for reclassification"""
    frmt = """
    Input array ... type {}
    {}\n
    Reclassification using
    :  from {}  
    :  to   {}
    :  mask is {} value is {}\n
    Reclassed array
    {}
    
    """
    a = np.arange(60).reshape((6,10))
    bins = [0,5,10,15,20,25,30,60,100]
    new_bins = [1, 2, 3, 4, 5, 6, 7, 8]
    mask = False
    mask_val = None
    a_rc = arr_reclass(a, bins=bins, new_bins=new_bins)
    args = [type(a).__name__, a, bins, new_bins, mask, mask_val, a_rc]
    print(dedent(frmt).format(*args))
    a_mask = np.ma.masked_where((a%7==0),a)
    mask = True
    mask_val = -1
    a_mask.set_fill_value(mask_val)
    a_mask_rc = arr_reclass(a_mask, bins=bins, new_bins=new_bins, mask=mask, mask_val=mask_val)
    args = [type(a_mask).__name__, a_mask, bins, new_bins, 
            mask, mask_val, a_mask_rc]
    print(dedent(frmt).format(*args))
    return a, a_rc, a_mask, a_mask_rc
    
if __name__=="__main__":
    """run demo"""
    a, a_rc, a_mask, a_mask_rc = demo()
    print("Arrays returned: a, a_rc, a_mask, a_mask_rc")
    
