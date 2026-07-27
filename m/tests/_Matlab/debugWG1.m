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
    Tt1 =  portT1(1:272,1:272);
    Tt2 =  portT2(1:266,1:266);
    [v1, e1] = eig(portT1\portS1);
    e1 = diag(e1);
    idx = find(abs(e1)>(2*pi/3e8*1e6));
    e1 = e1(idx);
    v1 = v1(:,idx);
    [e1,idx] = sort(real(sqrt(-e1))-imag(sqrt(-e1)),'descend');
    v1 = v1(:,idx);
    [v2, e2] = eig(portT2\portS2);
    e2 = diag(e2);
    idx = find(abs(e2)>(2*pi/3e8*1e6));
    e2 = e2(idx);
    v2 = v2(:,idx);
    [e2,idx] = sort(real(sqrt(-e2))-imag(sqrt(-e2)),'descend');
    v2 = v2(:,idx);
    matB(:,1) = vectorReader('mat_B0');
    matB(:,2) = vectorReader('mat_B1');
    matA = mmread('mat_A');
    matB = sparse(matB);
    matX = matA\matB;
    save WG
end

load WG;
k0 = 200;
z0 = 120*pi;

c0 = 299792458;
n = length(S);
idx = 1:n;
idx(Dir) = 0;
idx = find(idx);
S = S(idx,idx);
T = T(idx,idx);
G = zeros(2,n);
v1 = v1(1:272,1);
v2 = v2(1:266,1);
G(1,porti1) = e1(1)*Tt1*v1;
G(2,porti2) = e1(2)*Tt2*v2;
G = G(:,idx);

P = (G*G.') + 1i*k0*z0*eye(2);
% G = 1i*k0*z0*G;
A = sparse([P G; G.' S-40000*T]);
B = sparse(length(idx)+2,1);
B(1:2,1:2)= 1i*2*k0*z0*eye(2);
%B(3:end,1) = -2*G.'*modes;


X = A\B;
Sp = (full(X(1:2,1:2))-eye(2));
norm(Sp)
db(Sp)
