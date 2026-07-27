% test on efficient schur complement inversion

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

Aff0 = mmread('AFF0.mm');
Aff1 = mmread('AFF1.mm');
Aff2 = mmread('AFF2.mm');
Aff3 = mmread('AFF3.mm');
% figure; spy(Aff0)
% figure; spy(Aff1)
% figure; spy(Aff2)
% figure; spy(Aff3)


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

% figure;spy( triu(AFF0) - Aff0 - Aff1 - Aff2 - Aff3)
% beta = triu(AFF0) - Aff0 - Aff1;% - Aff2;% - Aff3;
% abs(beta) > 1e-11
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
TF = AFF0;
for ir=1:Nblocks-1
    fprintf('%g\n',length(AII{ir}));
    tic

    SF = SF - AFI{ir}*(AII{ir}\AFI{ir}.');
    Aii = diag(1./diag(AII{ir}));
    TF = TF - AFI{ir}*(Aii*AFI{ir}.');
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
P(idx(1)+1:idx(2),idx(2):end) = A(idx(1)+1:idx(2),idx(2):end);
%%
% opts.type = 'nofill'; opts.michol = 'on';
% L2 = ichol(SF,opts);
P(idx(1)+1:idx(2),idx(1)+1:idx(2)) = SF;

%%

Xdd = uF;
for i=2:Nblocks
    Xdd = [Xdd; uI{i-1}];
end
err = norm(full(X-Xdd))/norm(full(X));
fprintf('tot err = %g\n',err);

uF2 = B;
tic;
maxIter = 0;
for i=1:size(B,2)
    [uF2(:,i),flag,relres,iter,resvec]  = gmres(A,B(:,i),100,1e-12,length(A),P);
    maxIter = max(maxIter, length(resvec));
end
fprintf('%2.4g s nIter = %d\n', toc, maxIter);

% sp = 20*log10(abs(uF(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)
figure(10)
semilogy(1:length(resvec), resvec,'b');
hold on
% hold on
% load ILU
% semilogy(1:length(resvecILU), resvecILU,'.-r');
return

%%

% invAII = inv(AII{1});
% invAFF = inv(AFF0);
% Mat = AFI{1}*(invAII*(AFI{1}.'));
% Mat1 = Mat*0;
% Mat1(1,:) = Mat(1,:);
% Mat2 = Mat*0;
% Mat2(1,:) = Mat(1,:);
%  = invAFF *(1 + 1/trace(Mat1))*Mat1*invAFF
% uF

