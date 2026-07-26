function IOwVTK(Sys, Mesh, filename)
% Write a VTK file for electric field visualisation in Paraview.
tic;
fid = fopen(['FEpost/', filename, '.vtk'], 'w');
fprintf(fid, '# vtk DataFile Version 2.0\nUnstructured Grid\n');
fprintf(fid, 'ASCII\nDATASET UNSTRUCTURED_GRID\n');
fprintf(fid, 'POINTS %d double\n', length(Mesh.refNode));
for i = 1:length(Mesh.refNode)
    fprintf(fid, '%g %g 0\n', Mesh.refNode(i,1), Mesh.refNode(i,2));
end
fprintf(fid, 'CELLS %d %d\n', ...
    length(Mesh.refEle), 4*length(Mesh.refEle));
for i = 1:length(Mesh.refEle)
    fprintf(fid, '3 %d %d %d\n', ...
        Mesh.refEle(i,1)-1, Mesh.refEle(i,2)-1, Mesh.refEle(i,3)-1);
end
fprintf(fid, 'CELL_TYPES %d\n', length(Mesh.refEle));
for i = 1:length(Mesh.refEle)
    fprintf(fid, '9\n');
end
fprintf(fid, 'POINT_DATA %d\n', length(Mesh.refNode));

% Absolute value of electric field
fprintf(fid, 'SCALARS Eabs float 1\n');
fprintf(fid, 'LOOKUP_TABLE jet\n');
for i = 1:length(Mesh.refNode)
    fprintf(fid, '%f\n', abs(Sys.u(i)));
end

% Real part of electric field
fprintf(fid, 'SCALARS Ereal float 1\n');
fprintf(fid, 'LOOKUP_TABLE jet\n');
for i = 1:length(Mesh.refNode)
    fprintf(fid, '%f\n', real(Sys.u(i)));
end

% Imaginary part of electric field
fprintf(fid, 'SCALARS Eimag float 1\n');
fprintf(fid, 'LOOKUP_TABLE jet\n');
for i = 1:length(Mesh.refNode)
    fprintf(fid, '%f\n', imag(Sys.u(i)));
end

fclose(fid);
fprintf('Writing VTK: %2.4g s\n', toc);
