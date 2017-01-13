"""
Script:  pnts_timing
Author:  Dan.Patterson@carleton.ca
Modified: 2015-08-24
Purpose: 
    To provide some timing options on point creation and point-to-point distance calculations
    using einsum.
References:
    See these scripts for references:
        -
        -
Functions:
    decorators:  profile_func, timing, arg_deco
    main:  make_pnts, einsum_0
"""
import numpy as np
import random
import time
from functools import wraps

np.set_printoptions(edgeitems=5, linewidth=75, precision=2, suppress=True, threshold=5)


# .... wrapper funcs .............
def profile_func(func):
    """profile a function, input string as below
         prof_func('make_pnts()')
         prof_func('make_pnts(specify args/kwargs)')
    """
    import cProfile
    import pstats
    name = func
    stat_name = 'temp.txt'  # func + ".txt"
    cProfile.run(name,stat_name) 
    stats = pstats.Stats(stat_name)
    stats.strip_dirs().sort_stats('time').print_stats()  # time, tottime, cumtime

        
def timing(func):
    """timing decorator function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("\nTiming function for... {}".format(func.__name__))
        t0 = time.time()                # start time
        result = func(*args, **kwargs)  # ... run the function ...
        t1 = time.time()                # end time
        dt = t1-t0
        print("Time taken ...{} sec.".format(dt))
        #print("\n  print results inside wrapper or use <return> ... ")
        return result                   # return the result of the function
        #return dt                      # return delta time
    return wrapper


def arg_deco(func):
    """This wrapper just prints some basic function information."""
    @wraps(func)
    def wrapper(*args,**kwargs):
        print("Function... {}".format(func.__name__))
        #print("File....... {}".format(func.__code__.co_filename))
        print("  args.... {}\n  kwargs. {}".format(args,kwargs))
        return func(*args, **kwargs)
    return wrapper

   
# .... main funcs ................
@timing
@arg_deco
def make_pnts(x_min=0, x_max=1000, y_min=0, y_max=1000, num=1000):
    """Make (num) integer points between (low) and (high)."""
    dt = np.dtype([('ID', '<i4'), ('Shape', ('<f8', (2,)))]) 
    IDs = np.arange(0, num)
    Xs = np.random.random_integers(x_min, x_max, size=(num, 1))
    Ys = np.random.random_integers(y_min, y_max, size=(num, 1))
    arr = np.array(list(zip(IDs,zip(Xs, Ys))), dt)
    return arr


@timing
@arg_deco
def pnts_zeros(x_min=0, x_max=1000, y_min=0, y_max=1000, num=1000):
    """Produce ID, (X, Y) array using the zeros method"""
    Xs = np.random.random_integers(x_min, x_max, size=(num))  # ,1))
    Ys = np.random.random_integers(y_min,y_max,size=(num))    # ,1))
    IDs = np.arange(num)
    dt_3 = np.dtype([('ID', '<i4'), ('Shape',('<f8', (2,)))])
    zer = np.zeros(num,dt_3)
    zer['ID'] = IDs
    zer['Shape'][:,0] = Xs
    zer['Shape'][:,1] = Ys
    return zer


@timing
@arg_deco
def einsum_0(orig, dest):
    """einsum examples, see main script"""   
    diff = orig[:,None,:] - dest
    dist = np.sqrt(np.einsum('ijk, ijk->ij', diff, diff))
    print("\n(2) solution\n{}".format(dist))
    return dist

if __name__=="__main__":
    """time testing for various methods
    """
    #arr = make_pnts(x_min=0, x_max=1000, y_min=0, y_max=1000,num=1000)
    arr = pnts_zeros(x_min=0, x_max=1000, y_min=0, y_max=1000,num=1000)
    orig = arr['Shape'][0:2]
    dest = arr['Shape'][0:]
    d = einsum_0(orig,dest)
    # profile the function using either ...run or ...runctx
    #print("\nProfile output...")
    #profile_func('make_pnts()') # OR uncomment ....
    
    #cProfile.run('make_pnts()','make_pnts.txt')
    ##cProfile.runctx('make_pnts()',None,locals())
    #stats = pstats.Stats('make_pnts.profile')
    #stats.strip_dirs().sort_stats('time').print_stats()  # time, tottime, cumtime
  
