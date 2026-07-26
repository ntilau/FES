% test curls
clear
syms x y z;
p0 = 1-x-y-z;
p1 = x;
p2 = y;
p3 = z;

dp0 = gradient(p0,[x,y,z]);
dp1 = gradient(p1,[x,y,z]);
dp2 = gradient(p2,[x,y,z]);
dp3 = gradient(p3,[x,y,z]);

p0*dp1-p1*dp0
curl(p0*dp1-p1*dp0,[x,y,z])
