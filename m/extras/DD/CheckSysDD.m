clear all;% clc;

A = mmread('A.mm');
P = mmread('P.mm');
B = mmread('B.mm');

A = triu(A,0) + tril(A.',-1);
% spy(A)
err=1;
X = B*0;
while err > 1e-2
    Xn = A\( B - P*X);
    err = norm(full(Xn-X))/norm(full(Xn));
    X = Xn;
    fprintf('%g\n',err);
end


return
nRHS = size(B,2);

tic;
maxIter = 0;
for i=1:size(B,2)
    [X(:,i),flag,relres,iter,resvec]  = gmres(A+P,B(:,i),[],1e-12,size(A,1),A+tril(P));
    maxIter = max(maxIter, length(resvec));
%     pause
end
fprintf('%2.4g s nIter = %d\n', toc, maxIter);
figure; semilogy(1:length(resvec), resvec/max(resvec))

% return
% Xprev = full(B*0);
% err = 1;
% while err > 1e-6;
%     fprintf('error = %g\n', err);
%     X = A\(B-P*Xprev);
%     err = norm((X-Xprev))/norm(X);
%     Xprev = X;
%
% end
disp(20*log10(abs(X(1:nRHS,1:nRHS) - eye(nRHS))))
