# -*- coding: UTF-8 -*-
"""
:Script:   julia_set_mandelbrot.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-12-30
:Purpose:  Just because.
:
:Functions:  
:---------
:
:Notes:
:  varying the c parameter is the key
:  as is specifying a real zmin and zmax 
:  For g(z)=z^2−2 the Julia set is the line segment between −2 and 2.
:  See examples of Julia set equations here 
:  https://en.m.wikipedia.org/wiki/Julia_set
:    c=-0.8+0.156i  c=-0.70176-0.3842i  
:    z^2+(−0.7−0.3i)
:  golden ratio  = 1.6180339887....
:  c=(φ−2)+(φ−1)i =-0.4+0.6i
:Plotting:
:--------
:  imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, 
:         alpha=None, vmin=None, vmax=None, origin=None, extent=None,
:         shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None,
:         url=None, hold=None, **kwargs)
:  Display an image on the axes.
: - plot ranges in the order of -2 to 2 for both x and y
:
:Colors:
:------
:  
: nipy_spectral - black-purple-blue-green-yellow-red-white spectrum,
:                 originally from the Neuroimaging in Python project
#------------------------------------------------------------------------
: Values for constant c in z**2 + c where c is complex c =(real, img)
: scales near 2 work best ...
: c = (real, imag) complex(-0.7, 035) frost
: c=(0, 1)           dentrite fractal
:   -0.06 + 0.67j    Nice  try Spectral color
:   -0.065 + 0.66j
:   -0.123 + 0.745j  douady's rabbit fractal
:   -0.36  - 0.1     dragon
:   -0.391 - 0.587j  siegel disk fractal
:   -0.6565 + 0.4787878i    ocw example
:   -0.687 + 0.312j  cool one
:   -0.70 - 0.3j     NEAT cauliflower thingy
:   -0.70 + 0.35j    frost
:   -0.70176 - 0.3842i
:   -0.74543 + 0.11301j  really nice
:   -0.750 + 0j      san marco fractal
:   -0.75 - 0.2j     galaxies
:   -0.75 + 0.15j    groovy
:   -0.80 + 0.156j
:   -1.25 + 0.j      ocw example, horizontal feature
:   -1.815 + 0.0036j nice*** long skinny
: z**5 + c=0.544   * other powers, try z**3, 4, 5, etc
:
:References:
:----------
: matplotlib pyplot and colormaps
: - http://matplotlib.org/api/pyplot_summary.html
: - http://matplotlib.org/examples/color/colormaps_reference.html
:
: Julia and Mandelbrot sources:
: - https://www.researchgate.net/profile/Christian_Bauckhage/publication/
:       272679245_NumPy_SciPy_Recipes_for_Image_Processing_Creating_
: - Fractal_Images/links/54eb530e0cf2a0305193b9eb.pdf?origin=publication_list
: - http://paulbourke.net/fractals/juliaset/julia_set.py
: - http://kukuruku.co/hub/algorithms/julia-set ... for
:         c=-0.74543 + 0.11301j, c=-0.8+0.156j
: - http://ocw.nctu.edu.tw/upload/classbfs120905452520455.pdf
:        excellent explanation and photos,  c= -0.6565+0.4787878j, 
: - http://www.relativitybook.com/CoolStuff/julia_set.html 
:        has a map showing the parameter variation in the terms of c
: - https://www.linuxvoice.com/issues/010/julia.pdf  
:         good examples...
: - http://nbellowe.com/The%20Julia%20Set%20and%20Fractal/
:         Done in Julia language using 
:---------------------------------------------------------------------:
"""
#---- imports, formats, constants ----

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import dedent, indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.10f}'.format}
np.set_printoptions(edgeitems=3, linewidth=100, precision=10,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

#---- functions ----

def func(z, power=2, c=-0.74543 + 0.11301j):
    """Function used by julia.
    : power - normally 2, but can be higher orders
    : c - complex number... examples in documentation
    : c = complex(-0.1, 0.65)
    : default to use... c=-0.74543+0.11301j
    """
    return z**power + c

def julia(f, zmin, zmax, m, n, tmax=256):
    """Create Julia set images using numpy and python
    : f - the function to apply
    : zmin, zmax - z bounds as complex 
    : m, n - row, column image size
    : tmax - iteratio
    """
    xs = np.linspace(zmin.real, zmax.real, n)
    ys = np.linspace(zmin.imag, zmax.imag, m)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    J = np.ones(Z.shape) * tmax
    mask = np.abs(Z) <= 2.
    t = 0
    while (mask.sum() > 2) and t < tmax:
        if (t % 25) == 0:
            print("mask sum, {:>6.0f} t {:>3.0f}".format(mask.sum(), t))
        mask = np.abs(Z) <= 2.  # pythagorean distance squared based on -2,2
        Z[ mask] = f(Z[mask])
        J[-mask] -= 1
        t += 1
    return J, Z

  
def Mandelbrot(zmin, zmax, m, n, tmax=256):
    xs = np.linspace(zmin.real, zmax.real, n)
    ys = np.linspace(zmin.imag, zmax.imag, m)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    C = np.copy(Z)
    M= np.ones(Z.shape) * tmax
    for t in range(tmax):
        mask = np.abs(Z) <= 2.  # pythagorean distance squared based on -2,2
        Z[ mask] = Z[mask]**2 + C[mask]
        M[-mask] -= 1.
    return M

def julia_run():
    """Run the sample julia with defaults.  space can be changed to
    : complex(+/-2,+/-2) """
    zmin = complex(-0.025, -0.025)
    zmax = complex(0, 0)
    J, Z = julia(func, zmin, zmax, m=1000, n=1000, tmax=1024)
    cmap = cm.nipy_spectral 
    # cm.jet_r #cm.gist_rainbow #prism_r #cm.prism_r #cm.jet # cm.hot
    plt.imshow(J, cmap=cmap, origin="lower")
    plt.show()
    #name = 'julia3.png'
    #plt.imsave(name, J, cmap=cmap, origin='lower')
    frmt = """
    :Stats for julia run...
    : ranges (zmin, zmax) => {}, {}
    : julia (J.min, J.max) => {}, {}
    """
    print(dedent(frmt).format(zmin, zmax, J.min(), J.max()))
    return J, Z

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    J, Z = julia_run()


