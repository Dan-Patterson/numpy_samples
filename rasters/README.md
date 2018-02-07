**Raster array scripts**
------------------------

Not in any particular order... except diamond-square is just plain fun.

diamond_square.py
-----------------
algorithm is used for terrain generation... see DEM_stream.png for a sample of it being used to generate a surface for stream flow properties and other terrain derivates.

surface.py
----------
functions to calculate slope, aspect and hillshade.  I will try to find the rest of the options, but D8 is implemented ... just have to find the other options.

rasters.py
----------
Mostly functions to work with 3D stacks... aka, rasters for a location, and depth representing time.  Statistical functions for both masked (or arrays with nan) and non-masked arrays.  Plus a few others for stacking, reclassing.  I should reallyu make a table of contents ;)



