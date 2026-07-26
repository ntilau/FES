A = [1 0 0; 2 1 0; 3 6 4];
b = [1; 20; 2+1i*3];
% A = sprand(100,100,1);
% b = rand(100,1);
x = b*0;
M = eye(size(A));
restrt = 3;
max_it = length(b);
tol = 1e-6;
format long
%[x, error, iter, flag] = gmres2( A, x, b, M, restrt, max_it, tol );
[x1,flag1,relres1,iter1,resvec1] = gmres(A,b,restrt,tol,max_it);

norm(A\b-x1)
