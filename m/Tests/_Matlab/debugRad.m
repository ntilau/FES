% debug Radiation
addpath('Lib');
db = @(x,y) 20*log10(abs(x+1i*y));
disp(db(6.9310007717892752e-002,-3.3023005353849394e-001))
k0 = 2*pi*100e9/299792458.46775899438;
% A = mmread('A');
% K = sparse(6442,6442);
% rad = sparse(importdata('CoaxRadrad'));
% dof = sparse(importdata('CoaxRadradDof'))+2;
% K(dof,dof) = rad;
% K = 1i*k0*K;

% figure; spy(matA);
% figure; spy(K)
