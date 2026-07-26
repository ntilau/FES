%% boeing
nodes = importdata('nodes.txt');
faces = importdata('faces.txt');

trimesh(faces,nodes(:,1),nodes(:,2),nodes(:,3),nodes(:,3)*0);
axis equal; axis tight
colormap gray

polyID = fopen('boeing.poly','w');
fprintf(polyID, '#Nodes\n');
fprintf(polyID, '%d 3 0 0\n', length(nodes));
for i = 1:length(nodes)
    fprintf(polyID, '%d %e %e %e\n', i, nodes(i,1), nodes(i,2), nodes(i,3));
end
fprintf(polyID, '#Facets\n');
fprintf(polyID, '%d 1\n', length(faces));
%%% WP
% fprintf(polyID, '1 0 2\n');
% fprintf(polyID, '%d', N);
% for i = 1:N
%     fprintf(polyID, ' %d', i);
% end
% fprintf(polyID, '\n');
%%% PEC
for i=1:length(faces)
    fprintf(polyID, '1 0 1\n');
    fprintf(polyID, '3');
    fprintf(polyID, ' %d', faces(i,1));
    fprintf(polyID, ' %d', faces(i,2));
    fprintf(polyID, ' %d', faces(i,3));
    fprintf(polyID, '\n');
end

% return
% %%% RAD
% for i=Nz-2:Nz-1
%     for j=1:N-1
%         fprintf(polyID, '1 0 3\n');
%         fprintf(polyID, '4');
%         fprintf(polyID, ' %d', (i-1)*N+j);
%         fprintf(polyID, ' %d', i*N+j);
%         fprintf(polyID, ' %d', i*N+j+1);
%         fprintf(polyID, ' %d', (i-1)*N+j+1);
%         fprintf(polyID, '\n');
%     end
%     fprintf(polyID, '1 0 3\n');
%     fprintf(polyID, '4');
%     fprintf(polyID, ' %d', (i-1)*N+N);
%     fprintf(polyID, ' %d', i*N+N);
%     fprintf(polyID, ' %d', i*N+1);
%     fprintf(polyID, ' %d', (i-1)*N+1);
%     fprintf(polyID, '\n');
% end
% fprintf(polyID, '1 0 3\n');
% fprintf(polyID, '%d', N);
% for i = 1:N
%     fprintf(polyID, ' %d', (Nz-1)*N+i);
% end
fprintf(polyID, '\n');
fprintf(polyID, '#Holes\n');
fprintf(polyID, '0\n');
%fprintf(polyID, '1 0 0 0\n');
fprintf(polyID, '#Regions\n');
fprintf(polyID, '1\n');
dim = (max(nodes(:,3))-min(nodes(:,3)));
fprintf(polyID, '1 0 0 %g 0 %g\n',.5*dim, norm(nodes(:,1))/10);
fprintf(polyID, '#Solids 1\n');
fprintf(polyID, 'Air 1 1 0 0 vacuum\n');
fprintf(polyID, '#Boundaries 1\n');
fprintf(polyID, 'PEC1 1 PerfectE\n');
% fprintf(polyID, 'WP1 2 WavePort 2\n');
% fprintf(polyID, 'RAD 3 Radiation\n');
fclose(polyID);
