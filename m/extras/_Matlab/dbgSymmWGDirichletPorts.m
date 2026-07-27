%% full test
close all;
clear;
addpath('Lib')
format short;
db = @(x)20*log10(abs(x));
pi = 3.14159265359;
z0 = 3.767303134617707e3;
c0 = 299792458;
freq = linspace(5e9,25e9,61);%+rand(1,101)*1e6;
freq = 1.125e10;
nm = 1;
for ik = 1:length(freq)
    k0 = 2*pi*freq(ik)/c0;
    if false
        system(['ES WG ', num2str(freq(ik)), ' +verbose +lte']);
        Ssymm = mmread('S.mm');
        Tsymm = mmread('T.mm');
        Dir = importdata('DirEdges.dat');
        portS1 = importdata('portAWavePort1.dat');
        portT1 = importdata('portBWavePort1.dat');
        portS2 = importdata('portAWavePort2.dat');
        portT2 = importdata('portBWavePort2.dat');
        porti1 = importdata('EigVecDoFWavePort1.dat')+1;
        porti2 = importdata('EigVecDoFWavePort2.dat')+1;
        save SymmWGDir
%     pause(.2);
    else
        load SymmWGDir;
    end
    nm = 1;
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
%     e1(e1<0) = 0* e1(e1<0);
%     e2(e2<0) = 0* e2(e2<0);


%     clear idx dum;
    excit = eye(2*nm);
    n = length(S);
    idx = 1:n;
    idx(Dir) = 0;
    idx = find(idx);
    K = S-k0^2*T;
    portid = unique([porti1;porti2]');

    Tr = zeros(size(T));
    Tr(porti1,porti1) = Tt1;
    Tr(porti2,porti2) = Tt2;
    G1 = zeros(2*nm,n);
    G1(1:nm,porti1) = (v1(:,1:nm)).';
    G1(nm+(1:nm),porti2) =(v2(:,1:nm)).';
    for i=1:2*nm
        G1(i,:) = sqrt(G1(i,:)*Tr*G1(i,:).')\ G1(i,:);
    end
    nonPortid = 1:n;
    nonPortid(portid) = 0;
    nonPortid(Dir) = 0;
    nonPortid = nonPortid(nonPortid~=0);
    beta = diag([e1(1:nm); e2(1:nm)]);
    G1 = G1(:,portid);
    AII = K(nonPortid,nonPortid);
    AIT = K(nonPortid,portid);
    ATI = K(portid,nonPortid);
    ATT = K(portid,portid);

    ATIm = sparse(sqrt(beta)\G1*ATI*sqrt(k0*z0));
    ATTm = sparse(beta\((k0*z0)*G1*ATT*G1.'));
    ATTm = ATTm  + 1i*k0*z0*eye(2*nm);

    AR = [ATTm ATIm; ATIm.' AII];
    B = sparse(size(AR,1),2*nm);
    for i=1:2*nm
        B(i,i) = 2*1i*k0*z0;
    end

    %dirichlet
%     allDof = 1:n;
%     allDof(Dir) = -1;
%     allDof(portid) = 0;
%     allDof = allDof(allDof~=0);
%     allDof = find(allDof == -1)+2;
%     AR(allDof,:)=0;
%     AR(:,allDof)=0;
%     AR(allDof,allDof)= eye(length(allDof));


    X = AR\B;
    tmp = full(X(1:2*nm,1:2*nm))-eye(2*nm);
    Sp(:,ik) = tmp(:);
    nrm(ik) = norm(Sp(1:2*nm,ik));

%     format long
    fprintf('ES sol:\n')
    disp(db(tmp))
    disp(nrm(ik))
%     format short
%     disp([e1(1);e2(1)])
    gamma(:,ik) =  [e1(1:nm);e2(1:nm)];
%     disp(condest(AR))
% pause(.5)
%     end
end
if size(Sp,2)>1
    figure;
    plot(freq*1e-9, db(Sp))
end


return
matB(:,1) = vectorReader('mat_B0');
matB(:,2) = vectorReader('mat_B1');
matA = mmread('mat_A');
matB = sparse(matB);
matX = matA\matB;

fprintf('condition numbers:\n')
disp([condest(AR) condest(matA)])

matXSp = eye(2)-full(matX(1:2,1:2));
fprintf('LTE sol:\n')

disp(db(matXSp))

%
% figure; subplot(1,2,1); spy(AR); title('AR'); axis off;
% subplot(1,2,2); spy(matA); title('matA'); axis off;

disp(AR(1:2,1:2))
disp(matA(1:2,1:2))

row1 = AR(1,3:end);
row2 = matA(1,3:end);
