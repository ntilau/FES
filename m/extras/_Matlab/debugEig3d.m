clear;
addpath('Lib')
format long;

%
% St = mmread('Sor.mm');
% Tt = mmread('Tor.mm');

S = mmread('S.mm');
T = mmread('T.mm');
Dir = importdata('DirEdges.dat');
% save EV

% load EV
% load matsBig;
% save mats;
% load mats;
% full(St(6:-1:1,6:-1:1))
% full(S([5 6 3 4 1 2],[5 6 3 4 1 2]))
% return
% A = mmread('A');
% disp(full(S(1:6,1:6)))
% disp(full(A(1:6,1:6)))
% S = St;
% T = Tt;
% nnz(S)
% return;
c0 = 299792458;

% b=.1; a=.2; c=.4; m=1; n=0; p=1;
% kc = sqrt( (pi*m/a)^2 + (pi*n/b)^2 +(pi*p/c)^2);
% fcTE100 = c0/2/pi*kc;

n = length(S);
idx = 1:n;
idx(Dir) = 0;
idx = find(idx);
S = S(idx,idx);
T = T(idx,idx);

[vec, val] = eigs(S,T,20,'sa');
% [vec, val] = eig(full(S), full(T), 1);
val = diag(val);
% val = val(abs(val)>1e-3);
% var = sort(val);
val = sqrt(val)*c0/2/pi ;
val = val(abs(val)> 1e6);

% disp(fcTM110)
%%
disp(val(1:3).')
disp('7495274724.501030 10599076014.579599 15007502229.596399');
