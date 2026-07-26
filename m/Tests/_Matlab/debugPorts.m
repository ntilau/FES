% debug ports
clc; close all

totEdges = importdata('totEdges')+1;
totNodes = importdata('totNodes');
porti1 = importdata('EigVecDoFWavePort1.dat')+1;
porti2 = importdata('EigVecDoFWavePort2.dat')+1;


portS = importdata('portAWavePort1.dat');
portT = importdata('portBWavePort1.dat');
edges = importdata('edgesWavePort1.dat');
nodes = importdata('nodesWavePort1.dat');
%dof = importdata('dofEdgesWavePort1.dat');
dir = importdata('dirEdgesWavePort1.dat')+1;

% portS = importdata('portAWavePort2.dat');
% portT = importdata('portBWavePort2.dat');
% edges = importdata('edgesWavePort2.dat');
% nodes = importdata('nodesWavePort2.dat');
% dof = importdata('dofEdgesWavePort2.dat');
% dir = importdata('dirEdgesWavePort2.dat')+1;

% dir = importdata('portDir.dat') +1;

% portS = portS(dir,dir);
% portT = portT(dir,dir);

[v, e] = eig(portT\portS);
e = diag(e);
idx = find(abs(e)>(2*pi/3e8*1e6));
e = e(idx);
v = v(:,idx);
[e,idx] = sort(real(sqrt(-e))-imag(sqrt(-e)),'descend');
v = v(:,idx);
format shortEng;
disp(e(1))
v = v(1:length(idx),:);
% disp(v(:,1))
%norm(v(:,1))
% T = portT(1:length(idx),1:length(idx));

%%% plot sol
c = v(:,1);
edgev = totNodes(totEdges(:,1),:) + totNodes(totEdges(:,2),:);
edgev = edgev/2; % edge pos
edgen = -totNodes(totEdges(:,1),:) + totNodes(totEdges(:,2),:);
edgen = edgen(porti1,:);
edgev = edgev(porti1,:);
 figure; subplot(2,1,1);
quiver3(edgev(:,1), edgev(:,2), edgev(:,3), ...
    c.*edgen(:,1), c.*edgen(:,2), c.*edgen(:,3)); axis equal;  axis tight


edgev = nodes(edges(:,1),:) + nodes(edges(:,2),:);
edgev = edgev/2; % edge pos
edgen = -nodes(edges(:,1),:) + nodes(edges(:,2),:);
edgen = edgen(dir,[1 3]);
edgev = edgev(dir,[1 3]);
 subplot(2,1,2);
quiver(edgev(:,1), edgev(:,2),c.*edgen(:,1), c.*edgen(:,2)); axis equal; axis tight

return
% portS = importdata('portAWavePort1.dat');
% portT = importdata('portBWavePort1.dat');
% edges = importdata('edgesWavePort1.dat');
% nodes = importdata('nodesWavePort1.dat');
% dof = importdata('dofEdgesWavePort1.dat');
% dir = importdata('dirEdgesWavePort1.dat')+1;

portS = importdata('portAWavePort2.dat');
portT = importdata('portBWavePort2.dat');
edges = importdata('edgesWavePort2.dat');
nodes = importdata('nodesWavePort2.dat');
dof = importdata('dofEdgesWavePort2.dat');
dir = importdata('dirEdgesWavePort2.dat')+1;

% dir = importdata('portDir.dat') +1;

% portS = portS(dir,dir);
% portT = portT(dir,dir);

[v, e] = eig(portT\portS);
e = diag(e);
idx = find(abs(e)>(2*pi/3e8*1e6));
e = e(idx);
v = v(:,idx);
[e,idx] = sort(real(sqrt(-e))-imag(sqrt(-e)),'descend');
v = v(:,idx);
format shortEng;
disp(e(1))
v = v(1:length(idx),:);
% disp(v(:,1))
%norm(v(:,1))
T = portT(1:length(idx),1:length(idx));

%%% plot sol
edgev = nodes(edges(:,1),:) + nodes(edges(:,2),:);
edgev = edgev/2; % edge pos
edgen = -nodes(edges(:,1),:) + nodes(edges(:,2),:);
edgen = edgen(dir,[1 3]);
edgev = edgev(dir,[1 3]);
c = v(:,1); figure
quiver(edgev(:,1), edgev(:,2),c.*edgen(:,1), c.*edgen(:,2)); axis equal
