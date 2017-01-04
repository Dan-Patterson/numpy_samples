# -*- coding: UTF-8 -*-
"""
:Script:   ein_geom.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-12-08
:
:Purpose: To obtain polygon area or perimeter for multiple polygons.
:-------  Polygons may be simple, concave or convex.  They may be multipart
:  in nature, with either holes or disjoint parts.
:
:Functions:  help(<function name>) for help
:---------
:
:
:2D e_dist example:
:-----------------
:  2D data....
:  a = np.array([[ 0.,  0.], [ 1.,  1.], [ 2.,  2.], [ 3.,  3.], [ 4.,  4.]])
:  b = np.array([[ 0.,  0.], [ 1.,  0.], [ 2.,  0.], [ 3.,  0.], [ 4.,  0.]])
:  e_dist(a, b)
:  array([[ 0.000,  1.000,  2.000,  3.000,  4.000],
:         [ 1.414,  1.000,  1.414,  2.236,  3.162],
:         [ 2.828,  2.236,  2.000,  2.236,  2.828],
:         [ 4.243,  3.606,  3.162,  3.000,  3.162],
:         [ 5.657,  5.000,  4.472,  4.123,  4.000]])
:
:3d e_dist example:
:-----------------
:  a_3d = np.array([[ 0., 0., 0.], [ 1., 1., 1.], [ 2., 2., 2.], 
:                   [ 3., 3., 3.], [ 4., 4., 4.]])
:  b_3d = np.array([[ 0., 0., 0.], [ 1., 0., 0.], [ 2., 0., 0.],
:                   [ 3., 0., 0.], [ 4., 0., 0.]])
:  e_dist(a_3d, b_3d)
:  array([[ 0.     ,  1.     ,  2.     ,  3.     ,  4.     ],
:         [ 1.73205,  1.41421,  1.73205,  2.44949,  3.31662],
:         [ 3.4641 ,  3.     ,  2.82843,  3.     ,  3.4641 ],
:         [ 5.19615,  4.69042,  4.3589 ,  4.24264,  4.3589 ],
:         [ 6.9282 ,  6.40312,  6.     ,  5.74456,  5.65685]])
:  e_dist(a_3d, b_3d[0])
:  array([ 0.     ,  1.73205,  3.4641 ,  5.19615,  6.9282 ])
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
:---------------------------------------------------------------------:
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
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['e_area',
           'e_dist',
           'e_leng',
           'area_demo',
           'leng_demo',
           '_data']

#---- functions ----
def obj_area(obj):
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


def e_area(a, b=None):
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
    a = np.asarray(a)
    if isinstance(a, (list, tuple)):
        a = np.asarray(a)
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


def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...), cosine ('c','co'...)
    :
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim; b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1,1,a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]),1,a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]),b.shape[-1])
    diff =  a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def e_leng(a):
    """Length/distance between points in an array using einsum
    : Inputs
    :   a list/array coordinate pairs, with ndim = 3 and the 
    :   Minimum shape = (1,2,2), for example, (1,4,2) for a single line of 4 pairs
    :   The minimum input needed is a pair, but a sequence of pairs can be used.
    : Returns
    :   d_arr  the distances between points forming the array
    :   length the total length/distance formed by the points      
    """
    def cal(diff):
        """ perform the calculation
        :diff = g[:, :, 0:-1] - g[:, :, 1:]
        : for 4 d np.sum(np.sqrt(np.einsum('ijk...,ijk...->ijk...', diff, diff)).flatten())
        np.sum(np.sqrt(np.einsum('ijkl,ijkl->ijk', diff, diff)).flatten())
        """
        d_arr = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        length = np.sum(d_arr.flatten()) #np.sum(d_arr, axis=1)
        return length
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        return 0.0, 0.0
    if a.ndim == 2:
        a = np.reshape(a, (1,) + a.shape)
    if a.ndim == 3:
        diff = a[:, 0:-1] - a[:, 1:]  #a[:,0:-1] - a[:,1:]
        length = cal(diff)
    if a.ndim == 4:
        length = 0.0
        for i in range(a.shape[0]):
            diff = a[i][:, 0:-1] - a[i][:, 1:]
            length += cal(diff)
    return length

def poly2pnts(parts):
    """ ---- structured array creation ---- """    
    z = np.array([np.asarray(i) for i in parts])
    return z


def _data(dt='float64'):
    """Data used for area_demo and leng_demo demos
    """
    a = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]], dtype=dt)
    b = np.array([[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]], dtype=dt)
    c = -np.array(a)
    d = np.array([[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]], dtype=dt)
    e = np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]], dtype=dt)
    f = np.array([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]], dtype=dt)
    g = np.array([[a, d], [b, f]])
    h = np.array([a, c])
    a1 = [[0, 0], [0, 5], [0, 10], [5, 10],
          [10, 10], [10, 5], [10, 0], [5, 0], [0, 0]]  # densified
    b1 = [[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]]      # inner ring
    c1 = [[10, 0], [15, 10], [20, 0], [10, 0]]         # triangle
    v1 = [[a1, b1], c1]       # since the sizes aren't equal pass a list
    xy = [a, b, c, d, e, f, g, h, v1]
    return xy



def area_demo(dt='float64'):
    """Testing for separate polygons, those with holes and separate multipart
    : Example: 
    :  outer   x's: -10 - 10 = 20
    :          y's: -10 - 10 = 20 ... therefore area = 400
    :  inner   x's:  -5 -  5 = 10
    :          y's:  -5 -  5 = 10 ... therefore area = 100  ... final = 300    
    :
    """
    # outer (a,b,c)  inner (d,e,f)  holes( g)  multipart (h)
    def do_stuff(i, names, cnt):
        """ """
        args = [cnt, names[cnt-1], i.shape]
        frmt = "\n({}) Array\n {}... shape: {}"
        print(frmt.format(*args))
        print("\n{!r:}\narea {}\n{}".format(i, e_area(i), "-"*60))

    xy = _data()
    a, b, c, d, e, f, g, h, v1 = xy
    cnt = 1
    names = ["a_out", "b_out", "c_out",
             "d_in", "e_in", "f_in",
             "g_stacked holes [a,b], [d,f]",
             "h_separate_parts",
             "v_1 nested rings and triangle"]
    print("\nArea using... ein_area ...\n{}".format(area_demo.__doc__))
    for i in xy[:-1]:
        if isinstance(i, (list, tuple)):
            for ii in i:
                if len(ii)>1:
                    print("ii {}".format(ii[0]))
                ii = np.array(ii, dtype=dt)
                do_stuff(ii, names, cnt)
                cnt += 1
        else:
            do_stuff(i, names, cnt)
            #args = [cnt, names[cnt-1], i.shape]
        #frmt = "\n({}) Array\n {}... shape: {}"
        #print(frmt.format(*args))
        #print("\n{!r:}\narea {}\n{}".format(i, e_area(i), "-"*60))
            cnt += 1
    return None


def leng_demo(dt='float64'):
    """Determining perimeter to compliment the area calculations or the
    :  length or sequential points
    """
    xy = _data(dt)
    a, b, c, d, e, f, g, h, v1 = xy
    cnt = 1
    names = ["a_out", "b_out", "c_out",
             "d_in", "e_in", "f_in",
             "g_stacked holes [a,b], [d,f]", "h_separate_parts",
             "mixed 1"]
    print("\nArea using... ein_leng ...\n{}".format(leng_demo.__doc__))
    for i in xy:
        a = np.array(i)
        args = [cnt, names[cnt-1], a.shape]
        frmt = "\n({}) Array\n {}... shape: {}"
        print(frmt.format(*args))
        print("\n{!r:}\nlength {}\n{}".format(a, e_leng(a), "-"*60))
        cnt += 1
    return None   


def main():
    def obj_nd(b):
        ssub = []
        shp = b.shape[0]
        for i in range(shp):
            ssub.append(np.array(b[i]))
        return ssub
    a1 = [[0, 0], [0, 5], [0, 10], [5, 10],
          [10, 10], [10, 5], [10, 0], [5, 0], [0, 0]]  # densified
    b1 = [[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]]      # inner ring
    c1 = [[10, 0], [10, 10], [10, 0], [10, 0]]         # triangle
    a = np.asanyarray([[a1, b1], c1, [a1, b1, c1], b1])
    subs = []
    shp = a.shape[0]
    if (a.ndim == 1): # and (shp >= 1):
        for i in range(shp):
            sub = np.array(a[i])
            #print("before 1\n{}".format(sub))
            if sub.dtype.name == 'object':
                #print("before\n{}".format(sub))
                ssubs = obj_nd(sub)
                #print("after\n{}".format(ssubs))
                subs.extend(ssubs)
            else:
                s = np.array(sub)
                s = s.squeeze()
                subs.append(s)
        cnt = 0
        for i in range(len(subs)):
            val = e_leng(subs[i])
            print("({}): val: {}".format(cnt, val))
            cnt += 1
        print("---------")
    elif (a.ndim == 2):
        print(e_leng(a))
    #print(subs)    
    return a, a1, b1, c1, subs
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    a, b, c, d, e, f, g, h, v1 = _data(dt='float64')
    area_demo(dt='float64')
    #leng_demo(dt='float64')
    #a, a1, b1, c1, subs = main()
"""
a1 = [[0, 0], [0, 5], [0, 10], [5, 10],
          [10, 10], [10, 5], [10, 0], [5, 0], [0, 0]]  # densified
b1 = [[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]]      # inner ring
c1 = [[10, 0], [15, 10], [20, 0], [10, 0]]         # triangle
v1 = [[[a1, b1], c1]]       # since the sizes a
t = [a1, v1, b1, c1, v1]
bail = 10
x =[[i, np.array(t[i])] for i in range(len(t)) if len(t[i])>1]
"""
