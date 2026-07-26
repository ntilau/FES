%clc; clear all; close all;

% B = B(:,1);
A = mmread('A.mm');
B = mmread('B.mm');
P = mmread('P.mm');
% figure; spy(A)
% figure; spy(P)
% e = eigs(A\(A+P),50,'lm');
% figure; plot(e)

% A = (triu(A,0) + tril(A.',-1));
% X = A\B;
% fprintf('%2.4g s\n', toc);
% sp = 20*log10(abs(X(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)

% e = eig(full(A\A));
% k = 1;%(1e10*2*pi)/3e8;
%
% e = eig(full(((A)\(A-P))));
% figure(1)
% plot(e,'.')
% axis equal
% % axis([0 2 -1 1])
% return


idx = importdata('Blocks.txt');

fprintf('full, ');
A = (triu(A,0) + tril(A.',-1));
tic
X = A\B;
fprintf('%2.4g s\n', toc);
sp = 20*log10(abs(X(1:size(B,2),1:size(B,2))-eye(size(B,2))));
disp(sp)
% return

Nblocks = length(idx)-1;
% AFF = cell(Nblocks-2,1);
% for i=1:Nblocks-1
%     AFF{i} = mmread(['AFF',num2str(i-1),'.mm']);
%     AFF{i} = (triu(AFF{i},0) + tril(AFF{i}.',-1));
% end
%
% AII = cell(Nblocks-1,1);
% AFI = cell(Nblocks-1,1);
% f = cell(Nblocks-1,1);
% x = cell(Nblocks-1,1);
% xB = cell(Nblocks-1,1);
% x = cell(Nblocks-1,1);
AFF0 = A((idx(1)+1):idx(2),(idx(1)+1):idx(2));
% AFFsum = AFF0*0;
% for i=1:Nblocks-1
%     AFFsum = AFFsum + AFF{i};
% end
% fprintf('AFFerr = %g\n',norm(full(AFF0-AFFsum)));

% AFF0(3:end,3:end) = 0;
% AFF{1} = AFF{1} + AFF0/2;
% AFF{2} = AFF{2} + AFF0/2;
% % AFF0(1:2,:) = 0;
% % AFF0(:,1:2) = 0;
% return
%%
gF = B((idx(1)+1):idx(2),:);
% x0 = f0;
% rn = f0*0;
for i=2:Nblocks
    AII{i-1} = A((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1));
    AFI{i-1} = A((idx(1)+1):idx(2),(idx(i)+1):idx(i+1));
    f{i-1} = B((idx(i)+1):idx(i+1),:);
    %figure; spy(AFI{i-1}*(AII{i-1}\AFI{i-1}.'))
    %pause(1)
end

SF = AFF0;
for ir=1:Nblocks-1
    fprintf('%g\n',length(AII{ir}));
    tic

    SF = SF - AFI{ir}*(AII{ir}\AFI{ir}.');
    %AIIinv{ir} = AII{ir} - AFI{ir}.'*(AFF0\AFI{ir});
    toc
end
tic
uF = SF\gF;
toc
for ir=1:Nblocks-1
    fprintf('%g\n',length(AII{ir}));
    tic
    uI{ir} = -AII{ir}\(AFI{ir}.'*uF);
    toc
end

P = A*0;
P(idx(2):end,idx(2):end) = A(idx(2):end,idx(2):end);
P(idx(1)+1:idx(2),idx(1)+1:idx(2)) = SF;
P(idx(1)+1:idx(2),idx(2):end) = A(idx(1)+1:idx(2),idx(2):end);


Xdd = uF;
for i=2:Nblocks
    Xdd = [Xdd; uI{i-1}];
end
err = norm(full(X-Xdd))/norm(full(X));
fprintf('tot err = %g\n',err);

uF = B;
tic;
maxIter = 0;
for i=1:size(B,2)
    [uF(:,i),flag,relres,iter,resvec]  = gmres(A,B(:,i),100,1e-12,length(A),P);
    maxIter = max(maxIter, length(resvec));
end
fprintf('%2.4g s nIter = %d\n', toc, maxIter);

% sp = 20*log10(abs(uF(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)
figure(2)
semilogy(1:length(resvec), resvec,'b');
hold on
% hold on
% load ILU
% semilogy(1:length(resvecILU), resvecILU,'.-r');


return
% err = 1;
% while err > 0.1
for i=2:Nblocks
    x{i-1} = AII{i-1}\(f{i-1} - AFI{i-1}.'*x0);
    rn = rn + AFI{i-1}*x{i-1} + AFF{i-1}*x0 - f0;
    %xB{i-1} = AFF{i-1}\(- AFI{i-1}*x{i-1});
end
rnn = rn;
pn = rn;
pnn = rn;
qn = rn;
figure(1);
hold on;
for k=1:1000
    pn = pnn;
    rn = rnn;
    qn = qn*0;
    for i=2:Nblocks
        x{i-1} = AII{i-1}\(- AFI{i-1}.'*pn);
        qn = qn + AFI{i-1}*x{i-1} + AFF{i-1}*pn;
    end
    an = (rn.'*rn)/(pn.'*qn);
    x0 = x0 - an*pn;
    rnn = rn - an*qn;
    err = norm(rnn)/norm(rn);
    bn = (rnn.'*rnn)/(rn.'*rn);
%     fprintf('err = %g\n',err);
    pnn = rnn + bn*pn;
    figure(1); semilogy(k,err,'.'); axis tight
    %xB{i-1} = AFF{i-1}\(- AFI{i-1}*x{i-1});
end


Xdd = x0;
for i=2:Nblocks
    Xdd = [Xdd; x{i-1}];
end
err = norm(full(X-Xdd))/norm(full(X));
fprintf('tot err = %g\n',err);
% end


return
%
% xB = 0*gF;
% % xB = gF;
% xC = cell(Nblocks-1,1);
% for ir=1:Nblocks-1
% %     Atmp = AII{ir}\(AFI{ir}.');
%
% %     fprintf('dom %d, ', ir);
%     S = AII{ir} - AFI{ir}.'*( AFF{ir} \ AFI{ir} );
%     %xC{ir} = S\(- AFI{ir}.'*(AFF\gF));
%     xC{ir} = S\(- AFI{ir}.'*(AFF{ir}\gF));
%     xB = xB + AFF{ir}\(- AFI{ir}*xC{ir});
% %     fprintf('%2.4g s\n', toc);
% end
%
% sp = 20*log10(abs(xB(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)
%
%
%
return
% tic
% fprintf('full, ');
% A= (triu(A,0) + tril(A.',-1));
% X = A\B;
% fprintf('%2.4g s\n', toc);
% sp = 20*log10(abs(X(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)

P = spalloc(size(A,1), size(A,2),nnz(A));
Nblocks = length(idx)-1;
AII = cell(Nblocks-1,1);
AFI = cell(Nblocks-1,1);
AFF = A((idx(1)+1):idx(2),(idx(1)+1):idx(2));
P((idx(1)+1):idx(2),(idx(1)+1):idx(2)) = A((idx(1)+1):idx(2),(idx(1)+1):idx(2));
AFF = triu(AFF,0) + tril(AFF.',-1);
gF = B((idx(1)+1):idx(2),:);
for i=2:Nblocks
    AII{i-1} = A((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1));

    AFI{i-1} = A((idx(1)+1):idx(2),(idx(i)+1):idx(i+1));
    %P((idx(1)+1):idx(2),(idx(1)+1):idx(2)) = P((idx(1)+1):idx(2),(idx(1)+1):idx(2));% +...
   %     AFI{i-1}*(AII{i-1}\AFI{i-1}.');
    P((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1)) = ... A((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1)) -...
        AII{i-1};% + AFI{i-1}.'*( AFF \ AFI{i-1} );
    %AII{i-1} = triu(AII{i-1},0) + tril(AII{i-1}.',-1);
%     figure; spy(AII{i-1});
end


% aff = A(1:2,1:2);
% aii = A(3:end,3:end);
% afi = A(1:2,3:end);
% xB = aff\B(1:2,:);
% s = aii - afi.'*( aff \ afi );
% p = P(3:end,3:end);
% f = (- afi.'*xB);
% % tic
% % for i=1:Nblocks
% %     ind = (idx(i)+1):idx(i+1)-2;
% %     p(ind,ind) = s(ind,ind);
% %
% % end
% [L,U] = ilu(p,struct('type','ilutp','droptol',1e-12));
% % [L1,U1] = ilu(prec,struct('type','ilutp','droptol',1e-12));
% % L = L+tril(L1,-1);
% % [L,U] = lu(p,0.5);
% % fprintf('%2.4g s for precond LU\n', toc);
%
% xC = f*0;
% tic;
% maxIter = 0;
% for i=1:size(B,2)
%     [xC(:,i),flag,relres,iter,resvec]  = qmr(s,f(:,i),1e-9,1000);
%     maxIter = max(maxIter, length(resvec));
% end
% fprintf('%2.4g s nIter = %d\n', toc, maxIter);
% figure
% semilogy(1:length(resvec), resvec,'b');
% hold on
% load ILU
% semilogy(1:length(resvecILU), resvecILU,'.-r');
% %xC = s\f;
%
%
%
% xB = aff\(B(1:2,:) - afi*xC);
% sp = 20*log10(abs(xB(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)
%
% figure;spy(L*U)
% return

tic
[L,U] = ilu(P,struct('type','ilutp','droptol',1e-12));
%[L,U] = lu(P);
fprintf('%2.4g s for precond LU\n', toc);

uF = B;
tic;
maxIter = 0;
for i=1:size(B,2)
    [uF(:,i),flag,relres,iter,resvec]  = gmres(A,B(:,i),[],1e-9,1000,L,U);
    maxIter = max(maxIter, length(resvec));
end
fprintf('%2.4g s nIter = %d\n', toc, maxIter);

% sp = 20*log10(abs(uF(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)
figure
semilogy(1:length(resvec), resvec,'b');
hold on
load ILU
semilogy(1:length(resvecILU), resvecILU,'.-r');

% figure;spy(L*U)
return
%%
e = eigs(eye(size(A))-U\(L\A),30,'lr');
figure; plot(real(e),imag(e),'*', 1+cos(linspace(0,2*pi,101)),sin(linspace(0,2*pi,101)),'r')
axis equal
return
%Sref = mmread('S.mm');
%figure; spy(Sref);
%return
%AIIref = mmread('AII.mm');
% idx = importdata('Schur.txt');
% save matrices A B idx;
% load matrices
% P = mmread('P.mm');
% figure; spy(A);
% load matrices;


% return
% profile -memory on
%%
P = spalloc(size(A,1), size(A,2),nnz(A));
Nblocks = length(idx)-1;
AII = cell(Nblocks-1,1);
AFI = cell(Nblocks-1,1);
AFF = A((idx(1)+1):idx(2),(idx(1)+1):idx(2));
P((idx(1)+1):idx(2),(idx(1)+1):idx(2)) = A((idx(1)+1):idx(2),(idx(1)+1):idx(2));
AFF = triu(AFF,0) + tril(AFF.',-1);
gF = B((idx(1)+1):idx(2),:);
for i=2:Nblocks
    AII{i-1} = A((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1));
    P((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1)) = A((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1));
    AFI{i-1} = A((idx(1)+1):idx(2),(idx(i)+1):idx(i+1));
    AII{i-1} = triu(AII{i-1},0) + tril(AII{i-1}.',-1);
%     figure; spy(AII{i-1});
end

P = triu(P,0) + tril(P.',-1);
tic
%[L,U] = ilu(P,struct('type','nofill','droptol',1e-3));
[L,U] = lu(P);
fprintf('%2.4g s for precond LU\n', toc);
uF = B;
tic;
maxIter = 0;
for i=1:size(B,2)
    [uF(:,i),flag,relres,iter,resvec]  = gmres(A,B(:,i),[],1e-9,1000,L,U);
    maxIter = max(maxIter, length(resvec));
end
fprintf('%2.4g s nIter = %d\n', toc, maxIter);

sp = 20*log10(abs(uF(1:size(B,2),1:size(B,2))-eye(size(B,2))));
disp(sp)
 figure; semilogy(1:length(resvec), resvec)
return;
tic
xB = AFF\gF;
% xB = gF;
xC = cell(Nblocks-1,1);
for ir=1:Nblocks-1
%     Atmp = AII{ir}\(AFI{ir}.');

%     fprintf('dom %d, ', ir);
    S = AII{ir} - AFI{ir}.'*( AFF \ AFI{ir} );
    %xC{ir} = S\(- AFI{ir}.'*(AFF\gF));
    xC{ir} = S\(- AFI{ir}.'*xB);

%     fprintf('%2.4g s\n', toc);
end
xB = AFF\gF;
for ir=1:Nblocks-1
    xB = xB + AFF\(- AFI{ir}*xC{ir});
end
% for ir=1:Nblocks-1
% %     Atmp = AII{ir}\(AFI{ir}.');
%
% %     fprintf('dom %d, ', ir);
%     %S = AII{ir} - AFI{ir}.'*( AFF \ AFI{ir} );
%     %xC{ir} = S\(- AFI{ir}.'*(AFF\gF));
%     xC{ir} = AII{ir}\(- AFI{ir}.'*xB);
%
% %     fprintf('%2.4g s\n', toc);
% end
% % xB = AFF\gF;
% for ir=1:Nblocks-1
%     xB = xB - AFF\(- AFI{ir}*xC{ir});
% end
% Xs = [xB;xC];
% fprintf('Solerr = %e\n',norm(full(X-Xs)));
sp = 20*log10(abs(xB(1:size(B,2),1:size(B,2))-eye(size(B,2))));
disp(sp)

% return
%%
SF = AFF*0;
for ir=1:Nblocks-1
%     Atmp = AII{ir}\(AFI{ir}.');
    tic
    fprintf('dom %d, ', ir);
    SF = SF - AFI{ir}*(AII{ir}\(AFI{ir}.'));
    fprintf('%2.4g s\n', toc);
end
% norm(full(Atmp-AIIref))
tic
fprintf('bnd, ');
SF = SF+AFF;
% uF = SF\gF;
uF = gF;
[L,U] = ilu(SF,struct('type','ilutp','droptol',1e-4));
fprintf('%2.4g s ', toc);
tic;
maxIter = 0;
for i=1:size(gF,2)
    [uF(:,i),flag,relres,iter,resvec]  = qmr(SF,gF(:,i),1e-12,100,L,U);
    maxIter = max(maxIter, length(resvec));
end
fprintf('%2.4g s nIter = %d\n', toc, maxIter);
% figure; semilogy(1:length(resvec), resvec)
sp = 20*log10(abs(uF(1:size(B,2),1:size(B,2))-eye(size(B,2))));
disp(sp)
% fprintf('Error = %2.4e\n\n', norm(full(SF-Sref)));

% profile viewer
return
load old;
D = (Af)-(A);
figure; spy(D);
D = triu(Af)-triu(A);
figure; spy(D);
% norm(full(D))
Af(1:2,1:2)
A(1:2,1:2)
return
%%
b = ones(length(Af),1);
x1 = Af*b;
x2 = triu(Af,0)*b + tril(Af,-1)*b;
norm(x1-x2)

return

Af = A;
clear A
save old;


nRHS = size(B,2);
figure(1); spy(A);
figure(2); spy(P);


% X = (A-P)\B;
[uFapp,flag,relres,iter,resvec]  = gmres(SF,gF(:,1),[],1e-12,100);

% disp(20*log10(abs(X(1:nRHS,1:nRHS) - eye(nRHS))))

disp(20*log10(abs(X(1,1)-1)))
