% Test FETI-DP
%clc; clear all; close all;
idx = importdata('Blocks.txt');
Nblocks = length(idx)-1;

% B = B(:,1);
A = mmread('A.mm');
B = mmread('B.mm');
P = mmread('P.mm');
% figure; spy(A)
% figure; spy(P)
% e = eigs(A\(A+P),50,'lm');
% figure; plot(e)

A = (triu(A,0) + tril(A.',-1));
% X = A\B;
% fprintf('%2.4g s\n', toc);
% sp = 20*log10(abs(X(1:size(B,2),1:size(B,2))-eye(size(B,2))));
% disp(sp)
% gF = B((idx(1)+1):idx(2),:);
% x0 = f0;
% rn = f0*0;
for i=1:2
    AII{i} = A((idx(i)+1):idx(i+1),(idx(i)+1):idx(i+1));
    %figure; spy(AFI{i-1}*(AII{i-1}\AFI{i-1}.'))
    %pause(1)
end
ATC{1} = P((idx(1)+1):idx(2),(idx(2)+1):idx(3));
ATC{2} = P((idx(2)+1):idx(3),(idx(1)+1):idx(2));
G{1} = B((idx(1)+1):idx(2),1);
G{2} = B((idx(2)+1):idx(3),1);
X{1} = full(G{1}*0);
X{2} = full(G{2}*0);
%%
% err = 1;
% k = 1;
% while err>1e-6
%     x1 = AII{1}\(G{1} - ATC{1}*X{2});
%     x2 = AII{2}\(G{2} - ATC{2}*X{1});
%     err(k) = max(norm(x1-X{1})/norm(x1), norm(x2-X{2})/norm(x2));
%     fprintf('%g\n', err(k));
%     X{1} = x1;
%     X{2} = x2;
%     k = k+1;
% end
% semilogy(err)
% AIIinv{1} = inv(AII{1});
% AIIinv{2} = inv(AII{2});

% R
% AIIinv{1} = AIIinv{1}(8619:end,:);
% AIIinv{2} = AIIinv{1}(8143:end,:);
ATC{1} = ATC{1}(:,8143:end);
ATC{2} = ATC{2}(:,8619:end);
X{1} = X{1}(8619:end,:);
X{2} = X{2}(8143:end,:);
err = 1;
k = 1;
tic
while err>1e-12 & k<1000
    x1 = AII{1}\(G{1} - ATC{1}*X{2});
    x2 = AII{2}\(G{2} - ATC{2}*X{1});
    x1 = x1(8619:end,:);
    x2 = x2(8143:end,:);
    err(k) = max(norm(x1-X{1})/norm(x1), norm(x2-X{2})/norm(x2));
    t(k) = toc;
    fprintf('%g\n', err(k));
    X{1} = x1;
    X{2} = x2;
    k = k+1;
end
toc
semilogy(err)
