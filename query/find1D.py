# -*- coding: UTF-8 -*-
"""
:Script:   find1D.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-06-29
:Purpose:
:  Wrapper functions around 1D queries using np.where and
:  np.in1d, separately and in combination.
:Notes:
: finding stuff...
: - np.where is a builtin function
:   from numpy.core import multiarray as mu and buried in the multiarray.c code?
: really playing with boolean arrays if you examine the underlying code for np.in1d etc
: consider:
: - a = np.random.randint(0,10, size=10)
:     array([6, 6, 8, 6, 5, 7, 7, 0, 0, 4])
: - this = [4, 5, 6]
: - np.concatenate([a[a==i] for i in this])
:     array([4, 5, 6, 6, 6])
: - np.in1d(a, this)
:     array([1, 1, 0, 1, 1, 0, 0, 0, 0, 1], dtype=bool)
: - a[np.in1d(a,this)]
:     array([6, 6, 6, 5, 4])
: or.... Now note the differences in these two examples 
: - mask = np.zeros(len(a), dtype=np.bool)
: - for i in this:       # or... for i in this: mask |= (a==i)
:       mask |= (a == i)
: - mask
:     array([1, 1, 0, 1, 1, 0, 0, 0, 0, 1], dtype=bool)
: - for i in this:
:       mask &= (a!=i)
: - mask
:     array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0], dtype=bool)
: from the np.in1d code 
:
: Other stuff  Sequences
:    https://docs.python.org/3/library/stdtypes.html#typesseq
: double colon notation [start stop step ] with no commas
: - a = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
: - a[::2]    # a[start stop step] with no commas
:   array([ 0,  2,  4,  6,  8])
: - a[::-1]    # a[start stop step] with -1 reverses
:   array([9,  8,  7,  6,  5,  4,  3,  2, 1,  0])
: - a[::-3]    # a[start stop step] with -3 steps
:   array([9, 6, 3, 0])
: - a[::3]     #
:   array([0, 3, 6, 9])
:
: Now for 2D
: - a = np.arange(25).reshape(5,5)
:  array([[ 0,  1,  2,  3,  4],
:         [ 5,  6,  7,  8,  9],
:         [10, 11, 12, 13, 14],
:         [15, 16, 17, 18, 19],
:         [20, 21, 22, 23, 24]])
: - a[::2]               # skip every 2nd row
:  array([[ 0,  1,  2,  3,  4],
:         [10, 11, 12, 13, 14],
:         [20, 21, 22, 23, 24]])
: -  a[2::]              # start at row 2
:  array([[10, 11, 12, 13, 14],
:         [15, 16, 17, 18, 19],
:         [20, 21, 22, 23, 24]])
: -  a[:2:]              # first 2 rows
:  array([[0, 1, 2, 3, 4],
:         [5, 6, 7, 8, 9]])
: - a[::2,::2]           # now for both axes
:  array([[ 0,  2,  4],
:         [10, 12, 14],
:         [20, 22, 24]])
: - np.mean(a[::2,::2])         # 12.0
: - np.mean(a[::2,::2], axis=0) # array([ 10.000,  12.000,  14.000])
: - np.mean(a[::2,::2], axis=1) # array([ 2.000,  12.000,  22.000])
: without slicing
: - np.mean(a)        # 12.0
: - np.mean(a,axis=0) # array([ 10.000,  11.000,  12.000,  13.000,  14.000])
: - np.mean(a,axis=1) # array([ 2.000,  7.000,  12.000,  17.000,  22.000])
:
: - counting
: - a= np.array([1,2,3,1,2,1,2,3,4,1])
: - a.count(1)          # fails
: - a.tolist().count(1) # 4
:
:References
:
"""
#---- imports ----
import sys
import numpy as np
from textwrap import dedent
from timing import timing
#
#---- setup and constants ----
ft={'bool':lambda x: repr(x.astype('int32')),
    'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)
script = sys.argv[0]
#
#---- functions ----
@timing
def _func(fn, a, this):
    """called by 'find' see details there
    :  (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
    """
    #
    fn = fn.lower().strip()
    if fn in ['cumsum','csum', 'cu']:
        return np.where(np.cumsum(a) <= this)[0]
    if fn in ['eq','e','==']:
        return np.where(np.in1d(a, this))[0]
    if fn in ['neq', 'ne', '!=']:
        return np.where(~np.in1d(a, this))[0] #(a, this, invert=True)
    if fn in ['ls','les', '<']:
        return np.where(a < this)[0]
    if fn in ['lseq', 'lese', '<=']:
        return np.where(a <= this)[0]
    if fn in ['gt','grt', '>']:
        return np.where(a > this)[0]        
    if fn in ['gteq', 'gte', '>=']:
        return np.where(a >= this)[0]
    if fn in ['btwn', 'btw', '>a<']:
        low, upp = this
        return np.where( (a>=low) & (a<upp) )[0]
    if fn in ['btwni', 'btwi', '=>a<=']:
        low, upp = this
        return np.where( (a>=low) & (a<=upp) )[0]
    if fn in ['byond', 'bey', '<a>']:
        low, upp = this
        return np.where( (a<low) | (a>upp) )[0]

@timing     
def find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2):
    """ 
    : a    - array or array like
    : func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
    :        (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    : count - only used for recursive functions
    : keep - for future use
    : verbose - True for test printing
    : max_depth - to prevent recursive functions running wild, it can be varied
    :
    : recursive functions:
    : cumsum
    :   An example of using recursion to split a list/array of data
    :   parsing the results into groups that sum to this.  For example,
    :   split input into groups where the total population is less than
    :   a threshold (this).  The default is to use a sequential list,
    :   however, the inputs could be randomized prior to running.
    :
    :Returns
    :-------
    : a 1D or 2D array meeting the conditions
    :
    """
    a = np.asarray(a)              # ---- ensure array format
    this = np.asarray(this)
    if prn:                        # ---- optional print
        print("({}) Input values....\n  {}".format(count, a))
    ix = _func(func, a, this)      # ---- sub function -----
    if ix != None:
        keep.append( a[ix] )       # ---- slice and save
        if len(ix)>1:
            a = a[(len(ix)):]      # ---- use remainder
        else:
            a = a[(len(ix)+1):]
    if prn:                        # optional print
        print("  Remaining\n  {}".format(a))
    # ---- recursion functions check and calls ----
    if func in ['cumsum']:  # functions that support recursion
        if (len(a) > 0) and (count < r_lim): # recursive call
            count +=1
            find(a, func, this, count, keep, prn, r_lim)
        elif (count == r_lim):
            frmt = """Recursion check... count {} == {} recursion limit
              Warning...increase recursion limit, reduce sample size\n or changes conditions"""
            print(dedent(frmt).format(count, r_lim))
    # ---- end recursive functions ----
    #print("keep for {} : {}".format(func,keep))
    #
    if len(keep) == 1:   # for most functions, this is it
        final = keep[0]
    else:                # for recursive functions, there will be more
        temp = []
        incr = 0
        for i in keep:
            temp.append(np.vstack((i, np.array([incr]*len(i)))))
            incr +=1
        temp = (np.hstack(temp)).T
        dt = [('orig','<i8'),('class','<i8')]
        final = np.zeros((temp.shape[0],), dtype=dt)
        final['orig'] = temp[:,0]
        final['class'] = temp[:,1]
        # ---- end recursive section
    return final

def _demo():
    """Run the demo queries"""
    a = np.arange(20)
    count = 0
    r_lim = 30 # recursion limit
    #(cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
    tests = [['cumsum',20], ['eq',[2,5,7,10,15,18]],
             ['neq', [2,5,7,10,15,18]],
             ['ls', 15], ['lseq',15], ['gt', 15], ['gteq', 15],
             ['btwn', [8, 17]], ['btwni', [8, 17]], ['byond', [8, 17]]
             ]
    for test in tests:
        func, this = test
        final = find(a, func, this, count, keep=[], prn=False, r_lim= r_lim)
        print("\nFunction...: {}, this: {}\n  {}".format(func, this, final))   
    return a, final

@timing
def _large():
    a = np.random.randint(0,10, size=1000000) #
    final = find(a, 'byond',[2,7], 0, keep=[], prn=False, r_lim=200)
    print("Large test {} final\n{}".format(final.size,final))
    return a, final

if __name__=="__main__":
    """   """
    #a, final = _demo()
    #print("Script... {}".format(script))
    a, final = _large()
"""    N = 100000
    a = np.random.randint(0,5,size=N)
    this = 10000
    final = find(a, 'cumsum', this=this, count=0, keep=[], prn=False, r_lim=30)
"""
"""    a = np.array([6, 6, 8, 6, 5, 7, 7, 0, 0, 4])
    this = [4, 5, 6]
    mask = np.zeros(len(a), dtype=np.bool)
    for i in this:       # or... for i in this: mask |= (a==i)
        mask |= (a == i)
        print(mask)
"""    
    

