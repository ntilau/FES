% test WG
clear;
addpath('Lib')
format short;
db = @(x)20*log10(abs(x));

if false
S = mmread('S.mm');
T = mmread('T.mm');
Dir = importdata('DirEdges.dat');
portS1 = importdata('portAWavePort1.dat');
portT1 = importdata('portBWavePort1.dat');
portS2 = importdata('portAWavePort2.dat');
portT2 = importdata('portBWavePort2.dat');
porti1 = importdata('EigVecDoFWavePort1.dat')+1;
porti2 = importdata('EigVecDoFWavePort2.dat')+1;

[v1, e1] = eig(portT1\portS1);
e1 = diag(e1);
idx = find(abs(e1)>0);
e1 = e1(idx);
v1 = v1(:,idx);
[e1,idx] = sort(real(sqrt(-e1))-imag(sqrt(-e1)),'descend');
% e1 = sqrt(-e1(idx));
v1 = v1(1:length(idx),idx);
% v1(:,1)'*Tt1*v1(:,1)
% v1(:,2)'*Tt1*v1(:,1)

[v2, e2] = eig(portT2\portS2);
e2 = diag(e2);
idx = find(abs(e2)>0);
e2 = e2(idx);
v2 = v2(:,idx);
[e2,idx] = sort(real(sqrt(-e2))-imag(sqrt(-e2)),'descend');
% e2 = sqrt(-e2(idx));
v2 = v2(1:length(idx),idx);

Tt1 =  portT1(1:length(porti1),1:length(porti1));
Tt2 =  portT2(1:length(porti2),1:length(porti2));
% clear idx dum;
save WG
end

load WG;
c0 = 299792458;
k0 = 2*pi*1e10/c0;
z0 = 120*pi;

excit = [1 0]';

n = length(S);
idx = 1:n;
idx(Dir) = 0;
idx = find(idx);
S = S(idx,idx);
T = T(idx,idx);
G1 = zeros(2,n);
G1(1,porti1) = Tt1*v1(:,1);
G1(2,porti2) = Tt2*v2(:,1);
G1 = G1(:,idx);
D = (G1*G1.');
beta = 1i*diag([e1(1) e2(1)]);
A = sparse([D (1i*k0*z0)\G1; -1i*k0*z0*(beta*G1).' S-k0^2*T]);
% B = sparse(length(idx)+2,1);
% B(1:2,1:2)= -2*1i*k0*z0*P;
%B(3:end,1) = -2*G1.'*modes;
IE = -D*excit;
IH = -1i*k0*z0*(beta*G1).'*excit;
B = [IE;IH];


X = A\B;
Sp = full(X(1:2));
norm(Sp)
db(Sp)
