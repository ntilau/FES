function Mesh = IOrPoly(filename, args, hOrd, scal)
% Read a Triangle mesh generated from a .poly file via the IOrMesh binary.
%   filename : name of the .poly file (without extension)
%   args     : command-line arguments passed to IOrMesh
%   hOrd     : polynomial order for loading the .mat file
%   scal     : scaling factor for node coordinates
%
% The following args are enforced by IOrMesh:
%   -p  Triangulates a Planar Straight Line Graph (.poly file).
%   -D  Conforming Delaunay: all triangles are truly Delaunay.
%   -e  Generates an edge list.
%
% Additional args that can be passed through:
%   -q  Quality mesh generation (minimum angle may be specified).
%   -a  Applies a maximum triangle area constraint.
%   -u  Applies a user-defined triangle constraint.
%   -A  Applies attributes to identify triangles in certain regions.
%   -o2 Generates second-order subparametric elements.
tic
system(['cd FEpre/iormesh-src && ./IOrMesh ../', filename, ' ', args]);
load(['FEpre/', filename, '.h', num2str(hOrd), '.mat']);
Mesh.node = node * scal;
Mesh.ele = ele;
Mesh.spig = spig;
Mesh.spig2 = spig2;
Mesh.nlab = nlab;
Mesh.elab = elab;
Mesh.slab = slab;
Mesh.NNODE = size(node, 1);
Mesh.NELE = size(ele, 1);
Mesh.NSPIG = size(spig2, 1);
fprintf('Meshing geometry: %2.4g s\n', toc);
end
