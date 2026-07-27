clear all; clc
syms mx my mz hx hy hz M0 H0 g w;
linear = false;
Mdc = [0 0 M0].';
Hdc = [0 0 H0].';
if linear
    Mrf = [mx my 0].';
    Hrf = [hx hy 0].';
else
    Mrf = [mx my mz].';
    Hrf = [hx hy hz].';
end
Magnetization = Mdc + Mrf;
MagField = Hdc + Hrf;
Prod = g*cross(Magnetization,MagField);
Rhs = (1i*w)*[mx;my;mz];
w0 = g*H0;
wm = g*M0;
[A, b] = equationsToMatrix(Prod == Rhs, [mx, my, mz]);
X = A\b;
mu = collect(X + [hx;hy;hz],[hx;hy;hz]);
hz=0;
pretty(mu)
