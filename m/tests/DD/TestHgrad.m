% Hgrad p 2
syms x y z;
eta = 1-x-y-z;
a0 = eta*(2*eta-1);
jacobian(a0)
a1 = x*(2*x-1);
jacobian(a1)
