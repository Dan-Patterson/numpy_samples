# -*- coding: UTF-8 -*-
"""
:Script:   circle_make.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-01-02
:Purpose:
:-------
: - Make a circle or an arc
: - To form a polygon of an arc sector
:  top = _arc(radius=10, start=0, stop=1, step=0.2, xc=0.0, yc=0.0)
:  top.reverse()
:  bott = _arc(radius=9.5, start=0, stop=1, step=0.2, xc=0.0, yc=0.0)
:  top = np.array(top)
:  bott = np.array(bott)
:  close = top[0]
:  arc = np.asarray([i for i in [*top, *bott, close]])
:  pnts = arc.tolist()
:  pnts
:   [[9.99902524, 0.1396218], [9.99945169, 0.10471784],
:    [9.99975631, 0.0698126], [9.99993908, 0.03490651],
:    [10.0, 0.0], [9.5, 0.0], [9.49994212, 0.03316119],
:    [9.49976849, 0.06632197], [9.49947911, 0.09948195],
:    [9.49907398, 0.13264071], [9.99902524, 0.1396218]]
:Notes:
:
:References
:
"""
#---- imports, formats, constants ----
import sys
import numpy as np
from textwrap import dedent

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

ft={'bool': lambda x: repr(x.astype('int32')),
    'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)

script = sys.argv[0]

#---- functions ----
def plot_(pnts):
    """plot a circle, arc sector etc
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    #x_min = pnts[:,0].min()
    #x_max = pnts[:,0].max()
    #y_min = pnts[:,1].min()
    #y_max = pnts[:,1].max()
    fig, ax = plt.subplots()
    patches = []
    for i in pnts:
        polygon = Polygon(i, True)
        patches.append(polygon)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0)
    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))
    #ax.set_xlim(x_min-0.5, x_max+0.5)  # (x_min, x_max)
    #ax.set_ylim(y_min-0.5, y_max+0.5)  # y_min, y_max)
    ax.add_collection(p)
    plt.axis('equal')
    plt.show()
    plt.close()
    #return fig, ax


def rot_matrix(angle=0, nm_3=False):
    """Return the rotation matrix given points and rotation angle
    :Requires:
    :--------
    :  - rotation angle in degrees and whether the matrix will be used with
    :  - homogenous coordinates
    :Returns:
    :-------
    :  - rot_m - rotation matrix for 2D transform
    :  - rotate around  translate(-x, -y).rotate(theta).translate(x, y)
    """
    rad = np.deg2rad(angle)
    c = np.cos(rad)
    s = np.sin(rad)
    rm = np.array([[c, -s,  0.],
                   [s,  c,  0.],
                   [0., 0., 1.]])
    if not nm_3:
        rm = rm[:2,:2]
    return rm


def _arc(radius=100, start=0, stop=1, step=0.1, xc=0.0, yc=0.0):
    """
    :Requires:
    :---------
    :  radius = cirle radius from which the arc is obtained
    :  start, stop, incr = angles in degrees
    :  xc, yc - center coordinates in projected units
    :  
    :Returns: 
    :--------
    :  points on the arc
    """
    start, stop = sorted([start, stop])
    angle = np.deg2rad(np.arange(start, stop, step))
    Xs = radius*np.cos(angle)         # X values
    Ys = radius*np.sin(angle)         # Y values
    pnts = np.array(list(zip(Xs, Ys)))
    pnts = pnts + [xc, yc]
    p_lst = pnts.tolist()
    return p_lst


def _circle(radius=100, clockwise=True, theta=1, rot=0.0, scale=1, xc=0.0, yc=0.0):
    """   
    :Requires
    :--------
    :  radius - in projected units
    :  clockwise - True for clockwise (outer rings), 
    :            - False for counter-clockwise (for inner rings)
    :  theta -  angle spacing
    :     If theta=1, angles between -180 to < 180, are returned in 1 degree
    :     increments. The endpoint is excluded.
    :  
    :Returns
    :-------
    :  list of coordinates rounded to 3 decimal places
    :Notes:
    :------
    : You can also use np.linspace if you want to specify point numbers.
    : np.linspace(start, stop, num=50, endpoint=True, retstep=False)
    : np.linspace(-180, 180, num=720, endpoint=True, retstep=False) 
    """
    if clockwise:
        angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    else:
        angles = np.deg2rad(np.arange(-180.0, 180.0+theta, step=theta))
    #angles = np.append(angles, angles[0])
    x_s = radius*np.cos(angles)         # X values
    y_s = radius*np.sin(angles) * scale    # Y values
    pnts = np.c_[x_s,y_s]
    if rot != 0:
        rot_mat = rot_matrix(angle=rot) #angle)
        p_rot = np.dot(rot_mat, pnts.T) #p_trans.T).T + to_pnt
        pnts = p_rot.T + [xc, yc]
    return pnts


def arc_sector(outer=10, inner=9, start=1, stop=6, step=0.1):
    """ """
    top = _arc(radius=outer, start=start, stop=stop, step=step, xc=0.0, yc=0.0)
    top.reverse()
    bott = _arc(radius=inner, start=start, stop=stop, step=step, xc=0.0, yc=0.0)
    top = np.array(top)
    bott = np.array(bott)
    close = top[0]
    pnts = np.asarray([i for i in [*top, *bott, close]])
    #plot_(pnts)  # uncomment if using with plot_ function
    return pnts


def buffer_ring(outer_rad=100, inner_rad=0, theta=10, rot=0, scale=1, xc=0.0, yc=0.0):
    """Create a multi-ring buffer"""
    top = _circle(outer_rad, clockwise=True, theta=theta, rot=rot, scale=scale, xc=xc, yc=yc)
    if inner_rad != 0.0:
        bott = _circle(inner_rad, clockwise=False, theta=theta, rot=rot, scale=scale, xc=xc, yc=yc)
        pnts = np.asarray([i for i in [*top, *bott]])
    else:
        pnts = top
    #plot_(pnts)  # uncomment if using with plot_ function
    return pnts


def multiring_buffer():
    """Do a multiring buffer
    : rads - buffer radii
    : theta - angle density... 1 for 360 ngon, 120 for triangle
    : rot - rotation angle for ellipses and other shapes
    : scale - scale the y-values to produce ellipses
    """
    buffers = []
    rads = [ 10, 20, 40, 80, 100] #, 40, 60, 100]
    theta = 1
    rot = 22.5
    scale= 0.7
    clockwise=False
    for r in range(0, len(rads)):
        ring = buffer_ring(rads[r], rads[r-1], theta=theta, rot=rot, scale=scale)
        buffers.append(ring)
    plot_(buffers)
    #return buffers


def multi_sector():
    """ """
    sectors = []
    step = 5
    outer = 10
    inner = 9
    incr = np.arange(0, 91, 1) #(0,361,5)
    for outer in range(6, 10):
        inner = outer -1
        for i in range(1, len(incr)):
            st = incr[i]
            end = incr[i-1]
            arc = arc_sector(outer=outer, inner=inner, start=st, stop=end, step=0.1)
            sectors.append(arc)
    plot_(sectors)   


def help_():
    """ """
    print(__doc__)
#----------------------
if __name__=="__main__":
    """   """
    #print("Script... {}".format(script))
    #p_lst = _circle(radius=1, theta=30, xc=5, yc=5)
    #print(p_lst)
    #a_list = _arc(radius=10, start=0, stop=90.5, step=5, xc=0.0, yc=0.0)
    #print("\narc points...\n{}".format(a_list))
    #pnts = arc_sector()
    #pnts = buffer_rings()
    #multi_sector()
    multiring_buffer()
"""
rows = 4
cols = 6
N = 5    # your 51
a = np.arange(rows*cols).reshape(rows, cols)
b = np.arange(rows*cols).reshape(rows, cols) + 2
c = np.zeros_like(b)
print("Input arrays... a and b\n{}\n...\n{}".format(a, b))
for x in range (1, N):
    for i in range(0, 2):
        for j in range(0, 2):
            value1 = a[i, j]
            value2 = b[i, j]
            value1 = value1 * x
            value3 = value1 + value2
            c[i][j] = value3     # note change
        print("c at {}, {}\n{}".format(i, j, c))
"""
