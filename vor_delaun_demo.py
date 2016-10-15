# coding: utf-8
"""
Script:   vor_delaun_2016_06_25.py
Author:   Dan.Patterson@carleton.ca
Modified: 2016-06-01
References:
:Robust coordinates, orientation test etc
:- http://sixty-north.com/blog/the-folly-of-floating-point-for-robust-
:       geometric-computation
:- http://sixty-north.com/blog/rational-computational-geometry-in-python
:
:Collinear check and area calc from 3 points
:- http://stackoverflow.com/questions/3813681/checking-to-see-if-3-points-
:       are-on-the-same-line
:       
:Plot circle
- http://stackoverflow.com/questions/32092899/plot-equation-showing-a-circle
:    
: import matplotlib.tri as tri
:help(tri.delaunay)
: circumcenters, edges, tri_points, tri_neighbors = tri.delaunay(x, y)
: circumcenters -- shape-(numtri,2)
: edges -- shape-(nedges,2) array
: tri_points -- shape-(numtri,3)
: tri_neighbors -- shape-(numtri,3
:
: >>> Dd = tri.delaunay.delaunay(pnts[:,0],pnts[:,1])
: >>> Dd[0]   # circumcenter coordinates
: array([[ -0.0      ,  0.5       ],
:        [ 0.5       ,  1.0]])
: >>> Dd[1]
: array([[0, 4],
:        [4, 2],
:        [1, 2],
:        [0, 1],
:        [0, 3],
:        [0, 3],
:        [4, 1]], dtype=int32)
: >>> Dd[2]    # triangle point numbers
: array([[0, 4, 1],
:        [4, 2, 1]], dtype=int32)
: >>> Dd[3]    # triangle neighbors
: array([[ 1, -1, -1],
:        [-1,  0, -1]], dtype=int32)
:
: >>> D = tri.Triangulation(pnts[:,0],pnts[:,1])
: >>> D.edges  # point numbers connecting edges
: array([[0, 4],
:        [4, 2],
:        [1, 2],
:        [0, 1],
:        [4, 1]], dtype=int32)
: >>> D.neighbors
: array([[-1,  1, -1],
:        [-1, -1,  0]], dtype=int32)
: >>> D.triangles
: array([[0, 4, 1],
:        [4, 2, 1]], dtype=int32)
:
: check 
: matplotlib._delaunay.delaunay # used for python 3.5 and the new matplotib      
"""
import sys
import numpy as np
import matplotlib
import matplotlib._delaunay as dy
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.collections as LC
import time

def tri_area(a,b,c):
    """Triangle area sorted so that lengths of the sides. a >=b >= c
       0.25 * sqrt( (a+(b+c))*(c−(a−b))*(c+(a−b))*(a+(b−c)) )
       The parenthesis are not to be removed
    """
    t = [a,b,c]; t.sort()
    a,b,c = t
    area = 0.25 * np.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)) )
    return area
    

def vor(pnts, plot=True):
    """
    : D   delaunay triangulation
    : tr  triangles
    : C 
    """
    x = pnts[:,0]; y = pnts[:,1]
    lbl = np.arange(len(x))
    #D = tri.Triangulation(x, y) 
    #tr = D.triangles
    #n = tr.shape[0]
    #C = circumcircle2(pnts[tr])
    Dd = dy.delaunay(x, y)      # for version 3.5
    c_cents, edges, tri_pnts, tri_nbrs = Dd
    if plot:
        # Mask off unwanted triangles.
        min_radius = 0.25
        xmid = x[tri_pnts].mean(axis=1)
        ymid = y[tri_pnts].mean(axis=1)
        mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)  
        # Plot the triangulation.
        plt.figure()
        plt.gca().set_aspect('equal')
        #plt.triplot(D, 'bo-')
        for label, xpt, ypt in zip(lbl, x, y):
            plt.text(xpt, ypt, label)
        plt.triplot(x,y,tri_pnts, mask=mask)
        plt.scatter(x,y, s=20, c='b', marker='o')
        plt.scatter(c_cents[:,0], c_cents[:,1], s=40, c='r', marker='x')
        plt.title('triplot of Delaunay triangulation')
        plt.show()
        plt.close()
    return Dd, c_cents, edges, tri_pnts, tri_nbrs #D, tr, n, Dd

def v_plot(pnts, lines):
    """ """
    plt.scatter(pnts[:,0], pnts[:,1], color="blue")
    plt.title("Triangulation Visualization")
    plines = LC.LineCollection(lines, color='red')
    plt.gca().add_collection(plines)
    plt.axis('equal')
    #plt.axis((-20,120, -20,120))
    plt.axis ((-.2,1.2,-.2,1.2))
    plt.show()

def collinear(pnts):
    """ collinearity check, but also calculates area, if the area is 0, then
    : the points are collinear.  No need to divide by 2
    :  [ Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By) ] / 2
    """
    pnts = np.asarray(pnts)
    x0, y0, x1, y1, x2, y2 = pnts.ravel()
    a = (x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1))/2.0
    return a
    
def circ_3pnts(pnts):
    """Calculates the center and radius from 3 points.
    Input
    :pnts - a 2D array of x,y coordinates in planar coordinates
    Returns
    :x,y center in input coordinates
    :distance array representing the distance from each point to the center
    :a difference check consisting of 2 components
    Notes
    No checks are made since the errors in use will be obvious.
    Points are rolled out to their sequential x and y values, then squared
    for future use.  The equation for the center coordinates and the distance
    is calculated.
    """
    x0, y0, x1, y1, x2, y2 = pnts.ravel()
    x0s, y0s, x1s, y1s, x2s, y2s = np.square(pnts).ravel()
    x01, y01 = pnts[0]-pnts[1]
    x12, y12 = pnts[1]-pnts[2]
    d = (x01*y12) - (x12*y01)
    u = ((x0s - x1s) + (y0s - y1s))/2.0
    v = ((x1s - x2s) + (y1s - y2s))/2.0
    xc = (u*(y1-y2) - v*(y0-y1))/d
    yc = (v*(x0-x1) - u*(x1-x2))/d
    diff =  [xc,yc] - pnts
    dist_arr = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    dist = dist_arr[:-1]-dist_arr[1:]
    close = np.allclose(dist[0],dist[1])
    if close:
        c_dist = dist_arr[0]
    else:
        c_dist = c_dist.mean()
    return xc, yc, c_dist
        
if __name__=="__main__":
    """ """
    pnts = np.array([[0,0.],[0,1],[1,1],[1,0],[0.5,0.5]])
    Dd, c_cents, edges, tri_pnts, tri_nbrs = vor(pnts, True)
    print("Delaunay triangulation, using...\n{}".format(pnts))
    print("Edges...\n{}\nTriangle points...\n{}\n".format(edges, tri_pnts))
    print("Cirumcenter points..\n{}\nNeighbors...\n{}\n".format(c_cents, tri_nbrs))
    #lines, C, tr = voronoi2(pnts,(-.2, -.2, 1.2, 1.2))
    pnts2 = np.array([[0,0.],[0,1],[1,0]])
    xc, yc, c_dist = circ_3pnts(pnts2)
    print(xc, yc, c_dist)
    a = np.array([[6000,6000],[5999.999999,6000.0],[6000.001,6000.001]])
    b = np.array([[6000,6000],[6000., 5999.999999],[6000.001,6000.001]])
    print("collinear test a {}, b {}".format(collinear(a),collinear(b)))
    
