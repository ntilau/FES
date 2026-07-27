function WriteFileMeshMetis(ele, k)
% Write a .mesh file for partitioning with Metis.
%   ele : element-node connectivity matrix (nElements x 3 or x 4)
%   k   : 1 for triangles, 2 for tetrahedra
meshID = fopen('FileMeshMetis.mesh', 'w');

% First line: number of elements, number of subdomains
fprintf(meshID, '%d %d\n', length(ele), k);

% Remaining lines: element node indices
for i = 1:length(ele)
    fprintf(meshID, '%d %d %d\n', ele(i,1), ele(i,2), ele(i,3));
end

fclose(meshID);
