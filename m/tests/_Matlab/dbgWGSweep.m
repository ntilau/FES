%% full test
close all;
clear;
addpath('Lib')
format short;
db = @(x)20*log10(abs(x));
z0 = 120*pi;
c0 = 299792458;
freq = linspace(5e9,15e9,41);
freq = 11.25e9;

nm = 1;
for ik = 1:length(freq)
    k0 = 2*pi*freq(ik)/c0;
%     system(['ES WG ', num2str(freq(ik)), ' +verbose +lte']);
%     S = mmread('S.mm');
%     T = mmread('T.mm');
%     Dir = importdata('DirEdges.dat');
%     portS1 = importdata('portAWavePort1.dat');
%     portT1 = importdata('portBWavePort1.dat');
%     portS2 = importdata('portAWavePort2.dat');
%     portT2 = importdata('portBWavePort2.dat');
%     porti1 = importdata('EigVecDoFWavePort1.dat')+1;
%     porti2 = importdata('EigVecDoFWavePort2.dat')+1;
    load SymmWGDir;
    Tt1 =  portT1(1:length(porti1),1:length(porti1));
    Tt2 =  portT2(1:length(porti2),1:length(porti2));
    [v1, e1] = eig(portT1\portS1);
    e1 = diag(e1);
    idx = find(abs(e1)>0);
    e1 = e1(idx);
    v1 = v1(:,idx);
    [e1,idx] = sort(real(sqrt(-e1))-imag(sqrt(-e1)),'descend');
%     e1 = sqrt(-e1(idx));
    v1 = v1(1:length(idx),idx);
    % v1(:,1)'*Tt1*v1(:,1)
    % v1(:,2)'*Tt1*v1(:,1)

    [v2, e2] = eig(portT2\portS2);
    e2 = diag(e2);
    idx = find(abs(e2)>0);
    e2 = e2(idx);
    v2 = v2(:,idx);
    [e2,idx] = sort(real(sqrt(-e2))-imag(sqrt(-e2)),'descend');
%     e2 = sqrt(-e2(idx));
    v2 = v2(1:length(idx),idx);

%     v1(:,1) = sqrt(v1(:,1).'*Tt1*v1(:,1))\v1(:,1);
% %     v2(:,1) = sqrt(v2(:,1).'*Tt2*v2(:,2))\v2(:,1);
%     e1(e1>0) = -e1(e1>0);
%     e2(e2>0) = -e2(e2>0);
%     e1(e1<0) = e1(e1<0);
%     e2(e2<0) = e2(e2<0);


%     clear idx dum;
    excit = zeros(2*nm,1);
    excit(1) = 1;
    n = length(S);
    idx = 1:n;
    idx(Dir) = 0;
    idx = find(idx);
    K = S-k0^2*T;
%     portid = unique([porti1;porti2]');
%     K(portid,:) = 0;
%     K(:,portid) = 0;
%     K(portid,portid) = -eye(length(portid));
    K = K(idx,idx);
%     T = T(idx,idx);
    G1 = zeros(2*nm,n);
    G1(1:nm,porti1) = (v1(:,1:nm)).';
    G1(nm+(1:nm),porti2) =(v2(:,1:nm)).';
    Tr = zeros(size(T));
    Tr(porti1,porti1) = Tt1;
    Tr(porti2,porti2) = Tt2;
    G1 = G1(:,idx);%*Tr(idx,idx);
    Tr = Tr(idx,idx);
    for i=1:2*nm
        G1(i,:) = sqrt(G1(i,:)*Tr*G1(i,:).')\ G1(i,:);
    end
%     G1 = G1(:,idx)*T(idx,idx);
%     G1 = sqrt(G1*G1')\G1;
    beta = diag([e1(1:nm).' e2(1:nm).']);
    D = (G1*Tr*G1');
    P = (k0*z0)*(beta*G1*Tr).';
    Q = (1i*k0*z0)\(G1*Tr);
    A = sparse([D Q; P K]);
    IE = -D*excit;
    IH = P*excit;
    B = [IE;IH];
    X = A\B;
    Sp(:,ik) = full(X(1:2*nm));
    disp(db(Sp(:,ik)))
    nrm(ik) = norm(Sp(:,ik));
    disp(nrm(ik))
    disp(condest(A))
%
%     edges = importdata('edgesWavePort1.dat');
% nodes = importdata('nodesWavePort1.dat');
% %dof = importdata('dofEdgesWavePort1.dat');
% dir = importdata('dirEdgesWavePort1.dat')+1;
%     edgev = nodes(edges(:,1),:) + nodes(edges(:,2),:);
% edgev = edgev/2; % edge pos
% edgen = -nodes(edges(:,1),:) + nodes(edges(:,2),:);
% edgen = edgen(dir,[1 3]);
% edgev = edgev(dir,[1 3]);
% c = v1(:,1); figure(1); subplot(2,1,1)
% quiver(edgev(:,1), edgev(:,2),c.*edgen(:,1), c.*edgen(:,2)); axis equal
%     edges = importdata('edgesWavePort2.dat');
% nodes = importdata('nodesWavePort2.dat');
% %dof = importdata('dofEdgesWavePort1.dat');
% dir = importdata('dirEdgesWavePort2.dat')+1;
%     edgev = nodes(edges(:,1),:) + nodes(edges(:,2),:);
% edgev = edgev/2; % edge pos
% edgen = -nodes(edges(:,1),:) + nodes(edges(:,2),:);
% edgen = edgen(dir,[1 3]);
% edgev = edgev(dir,[1 3]);
% c = v2(:,1);
% subplot(2,1,2)
% quiver(edgev(:,1), edgev(:,2),c.*edgen(:,1), c.*edgen(:,2)); axis equal
end
if size(Sp,2)>1
figure;
plot(freq/1e9, db(Sp));
axis([ 5 15 -35 0]); axis tight
end

return
%%
beta = diag([e1(1) e2(1)]);
K = (S-k0^2*T);
D = ((beta*G1)*K*(beta*G1).')+(1i*k0*z0)*eye(2);


    A = sparse([D (k0*z0)*(beta*G1); (k0*z0)*(beta*G1).' K]);
    B = sparse(length(idx)+2,1);
    B(1:2,1)= 2*(k0*z0)*(D)*[0 1].';
    %B(3:end,1) = -2*G1.'*modes;
%     IE = ;
%     IH = (beta*G1).'*excit;
%     B = [IE;IH];


    X = A\B;
    Sp(:,ik) = full(X(1:2)) - excit;
    disp(db(Sp))
        disp(norm(Sp(:,ik)))

return

plot(freq, db(Sp))


% matB(:,1) = vectorReader('mat_B0');
% matB(:,2) = vectorReader('mat_B1');
% matA = mmread('mat_A');
% matB = sparse(matB);
% matX = matA\matB;
%
% matXSp = eye(2)-full(matX(1:2,1:2));
% db(matXSp)
