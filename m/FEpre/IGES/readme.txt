

                   IGES TOOLBOX FOR MATLAB
                   =======================


This is a version of the IGES toolbox for MATLAB. It can be downloaded at the Mathworks file exchange community site,

http://www.mathworks.com/matlabcentral/fileexchange/13253-iges-toolbox

The toolbox cannot handle all IGES entity types but I hope it will be of great help for you to develop your applications.
To add more entities in iges2matlab, read and understand the IGES specification,

www.uspro.org/documents/IGES5-3_forDownload.pdf

If you would like to share your upgraded version with other, please send it to me, per.bergstrom@ltu.se.
I can stronly recommend you to use I-DEAS IGES translator to generate the IGES-file to be read into Matlab.

Run some of the example files; "example", "example2", "exampleProjection", ...
in MATLAB's Command window for examples. Unfortunately, no good documentation of the functions is done due to lack of time,
but help yourself with the attached examples.

In this version some mex source files are included. Compile them in MATLAB by running "makeIGESmex" in the Command window.
All users must first compile the source-code before they can use it. See "help mex" in MATLAB for more information.


Parts of the code in Mesh2d v2.3, written by Darren Engwirda,

http://www.mathworks.com/matlabcentral/fileexchange/25555-mesh2d-automatic-mesh-generation

is embedded. It is not used by default, but in case of specific requirements it can easily be used.


                 MAIN FUNCTIONS
                 ==============


iges2matlab
-----------

This is the main function which extracts IGES parameter data from a file into Matlab. It cannot handle all
IGES-entity types. If the surface of the object is not correct, try with setting the flag "useTransformationEnityArb" to 0.


plotIGES
--------

Plots lines, curves, points and surfaces of an IGES-object.


plotIGESentity
--------------

Plots a given entity of an IGES-object.


plotNURBS
---------

Plots a NURBS entity (surface or curve).


transformIGES
-------------

Transform the parameter data of an IGES-object with a rotation/reflection and a translation.


findEntityIGES
--------------

Returns a vector with indices having a given entity type number.


projIGES
--------

Returns points of projections on a surface of an IGES-object.


projpartIGES
------------

Returns points of projections on part of a surface of an IGES-object.


projpartPerspectiveIGES
-----------------------

Returns points of perspective projections on part of a surface of an IGES-object.


projpartSphericalIGES
---------------------

Returns points of spherical projections on part of a surface of an IGES-object.


srfRepInProjection
------------------

Returns a surface representation using input/output from projpartIGES. See "exampleProjection2", and "exampleProjection2b".


linAppSrfIGES
-------------

Returns a surface approximation.


icpSrfLinRep
------------

Returns a surface representation.


getDVGtree
----------

Returns a DVG tree.


stl2matlab
----------

Extracts data in STL-files to Matlab.


plotSTL
-------

Plots data from STL-files.


transformSTL
------------

Transform vertices of a STL-object with a rotation/reflection and a translation.


igesToolBoxGUI
--------------

Matlab GUI. (igesToolBoxGUI.m and igesToolBoxGUI.fig)


                  SUBFUNCTIONS
                  ============


retSrfCrvPnt
------------

Returns values from surfaces, curves and points. No complete documentation is given.


nrbDerivativesIGES
------------------

Returns first and second derivative representation of NURBS.


nrbCrvPlaneIntrsctIGES (mex function)
-------------------------------------

Finds a parameter value of a NURBS curve at the closest point to a given plane.


nrbevalIGES (mex function)
--------------------------

Evaluates NURBS and derivatives of NURBS.


nrbSrfRegularEvalIGES (mex function)
------------------------------------

Evaluates a NURBS surface at parameter values in a regular grid.


closestNrbLinePointIGES (mex function)
--------------------------------------

Returns the closest point to a NURBS patch and a line/point.


createDVGtree (mex function)
----------------------------

Creates a DVG-tree.


icpDVG (mex function)
---------------------

An ICP algorithm using the DVG-tree.


makeIGESmex
-----------

m-file for compiling the mex files.


For more documentation about the functions above, see the help for each function in Matlab.
The folder mexSourceFiles contains source code for subfunctions of the mex-files.


Do you like this toolbox and have use of it? Please, let me know that.


/ Per Bergström     ( per.bergstrom at ltu.se )

