%% full test
close all;
clear;
addpath('Lib')
format short;
db = @(x)20*log10(abs(x));

k0 = 2*pi*1e10/299792458.46775899438;
S = mmread('S.mm');
T = mmread('T.mm');
% matB(:,1) = vectorReader('mat_B0');
% matB(:,2) = vectorReader('mat_B1');
matA = mmread('mat_A');
matA = matA(3:end,3:end);
A = S-k0^2*T;
% matB = sparse(matB);
% matX = matA\matB;
% matXSp = eye(2)-full(matX(1:2,1:2));
% db(matXSp)
idx = 0*6+(1:6);
full(A(idx,idx))
full(matA(idx,idx))
