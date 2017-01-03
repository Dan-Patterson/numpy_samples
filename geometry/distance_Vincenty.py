'''
distance_Vincenty.py

Author:  dan.patterson@carleton.ca  (code structure modified technically)

Modified:  Oct 2014

Purpose:
  Calculates the Vincenty Inverse distance solution for 2 long/lat pairs

Source:
  http://www.movable-type.co.uk/scripts/latlong-vincenty.html  java code
From:
  T Vincenty, 1975 "Direct and Inverse Solutions of Geodesics on the
  Ellipsoid with application of nested equations", Survey Review, vol XXIII
  no 176, 1975
  http://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf

Notes:
  atan2(y,x) or atan2(sin, cos) not like Excel
  used fmod(x,y) to get the modulous as per python

Returns:
  distance in meters and the initial and final bearings (as an azimuth from N)

Tips:
  import all of the 'math' module so you can access all the functions simply

'''

from math import *

def getDistance(long0,lat0, long1, lat1):
  '''return the distance on the ellipsoid between two points using
     Vincenty's Inverse method'''

  a = 6378137.0;  b = 6356752.314245
  f = 1 / 298.257223563

  L = radians(long1 - long0)

  U1 = atan((1 - f) * tan(radians(lat0)));   #check these
  U2 = atan((1 - f) * tan(radians(lat1)));
  sinU1 = sin(U1);	cosU1 = cos(U1)
  sinU2 = sin(U2);	cosU2 = cos(U2)          #down to here
  cosSqAlpha = float()
  sinSigma = float()
  cos2SigmaM = float()
  cosSigma = float()
  sigma = float()

  # l == lambda
  delta_long = L               #  first approximation
  lambdaP  = float()
  iterLimit = 100
  while (iterLimit > 0):
    sinLambda = sin(delta_long);  cosLambda = cos(delta_long)
    sinSigma = sqrt((cosU2*sinLambda)**2  + (cosU1*sinU2 - sinU1*cosU2*cosLambda)**2)  # eq 14
    if (sinSigma == 0):
      return 0;

    cosSigma = sinU1*sinU2 + cosU1*cosU2*cosLambda          # eq 15
    sigma = atan2(sinSigma, cosSigma)                       # eq 16   #check atan2 implementation
    sinAlpha = cosU1*cosU2*sinLambda/sinSigma               # eq 17
    cosSqAlpha = 1 - sinAlpha**2
    cos2SigmaM = cosSigma - 2*sinU1*sinU2/cosSqAlpha        # eq 18

    C = f/16.0 * cosSqAlpha*(4 + f*(4 - 3*cosSqAlpha))      # eq 10
    lambdaP = delta_long
    delta_long = L + (1 - C)*f*sinAlpha*(sigma + C*sinSigma*(cos2SigmaM + C*cosSigma*(-1.0 + 2*cos2SigmaM**2)))

    if (iterLimit == 0):          #is it time to bail?
      return 0.0
    elif((fabs(delta_long - lambdaP) > 1e-12) and (iterLimit > 0)):
      iterLimit = iterLimit - 1
    else:
      break

  uSq = cosSqAlpha * (a**2 - b**2)/b**2
  A = 1 + uSq/16384.0	* (4096 + uSq*(-768 + uSq*(320 - 175*uSq)))
  B = uSq/1024.0 * (256 + uSq*(-128 + uSq*(74 - 47*uSq)))
  deltaSigma = B*sinSigma*(cos2SigmaM + (B/4.0)*(cosSigma*(-1 + 2*cos2SigmaM**2) - (B/6.0)*cos2SigmaM*(-3 + 4*sinSigma**2)*(-3 + 4*cos2SigmaM**2)))

  dist = b*A*(sigma - deltaSigma)   # distance
  # from and to bearings in Java
  alpha1 = atan2(cosU2*sinLambda,  cosU1*sinU2-sinU1*cosU2*cosLambda)
  alpha2 = atan2(cosU1*sinLambda, -sinU1*cosU2+cosU1*sinU2*cosLambda)

  alpha1 = degrees(fmod((alpha1 + 2*pi),(2*pi))) # normalise to 0...360  degrees
  alpha2 = degrees(fmod((alpha2 + 2*pi),(2*pi))) # normalise to 0...360  degrees
  return dist,alpha1,alpha2

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
  #format getDistance(long0,lat0, long1, lat1)     dist (m.mmm) initial    final (ddd.ddd)
##  print getDistance(-75.0, 45.0, -75.0, 46.0) #  N  111141.548,   0.000,   0.000  values
##  print getDistance(-75.0, 46.0, -75.0, 45.0) #  S  111141.548, 180.000, 180.000  truncated
##  print getDistance(-76.0, 45.0, -75.0, 45.0) #  E   78846.334,  89.646,  90.353  to 3 dec.
##  print getDistance(-75.0, 45.0, -76.0, 45.0) #  W   78846.334, 270.353, 269.646  places
##  print getDistance(-76.0, 46.0, -75.0, 45.0) # SE  135869.091, 144.526, 145.239
##  print getDistance(-75.0, 46.0, -76.0, 45.0) # SW  135869.091, 215.473, 214.760
##  print getDistance(-76.0, 45.0, -75.0, 46.0) # NE  135869.091,  34.760,  35.473
##  print getDistance(-75.0, 45.0, -76.0, 46.0) # NW  135869.091, 325.239, 324.526

## ----- uncomment the above ----- or use below  --------------------
  coord = [-76.0, 46.0, -75.0, 45.0] # SE
  a0,a1,a2,a3 = coord
  b0, b1, b2 = getDistance(a0, a1, a2, a3)
  print ("\nFrom/to (long, lat) ({0}, {1}) : ({2}, {3})".format(a0,a1,a2,a3) )
  print ("  Dist {0:>15},  Initial {1:>15},  Final {2:>15}".format(b0,b1,b2) )