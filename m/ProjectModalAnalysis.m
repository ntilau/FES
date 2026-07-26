%% ProjectModalAnalysis - Compute TE/TM cutoff frequencies and dispersion of a rectangular waveguide
clear all;
close all
Config()

%% Setup geometry and polynomial order
a = 0.9 * 25.4;
b = 0.4 * 25.4;
epsr = 1;
Sys.pOrd = 3;
Sys.hOrd = 1;
TM11 = false;
prjName = 'ModelWR90';

Mesh = BuildRegularSquare(5, 3);
Mesh.node = Mesh.node * [a 0; 0 b] * 1e-3;
Mesh.BC.Dir = 1;

Sys.Hcurl = 1;
[Sys, Mesh] = AssembLin(Sys, Mesh);

%% Build reduced system (remove Dirichlet DOFs)
idx = 1:length(Sys.T);
idx(Sys.Dir{1}) = 0;
idx = find(idx);

Tte = Sys.T;
Ste = Sys.S;

%% Solve eigenvalue problem for cutoff wavenumbers
scal = 1;

[e] = eigs(Ste, epsr * Tte, 2, (2 * pi * 5e9 / 3e8) ^ 2);
e = e(abs(e) > 1e-5);
val0 = sqrt(e(1)) * 3e8 / 2 / pi

%% Second formulation using transverse/longitudinal splitting
if true
    if Sys.pOrd == 1
        dir = find(~Mesh.slab);
    elseif Sys.pOrd == 2
        dir = [find(~Mesh.slab); ...
            Mesh.NSPIG + find(~Mesh.slab); ...
            (2 * Mesh.NSPIG + 1:2 * Mesh.NSPIG + 2 * Mesh.NELE).'];
    elseif Sys.pOrd == 3
        dir = [find(~Mesh.slab); ...
            Mesh.NSPIG + find(~Mesh.slab); ...
            (2 * Mesh.NSPIG + 1:2 * Mesh.NSPIG + 2 * Mesh.NELE).'; ...
            2 * Mesh.NSPIG + 2 * Mesh.NELE + find(~Mesh.slab); ...
            (3 * Mesh.NSPIG + 2 * Mesh.NELE + 1:3 * Mesh.NSPIG + 6 * Mesh.NELE).'];
    end
    dirn = 1:Sys.NDOFs;
    dirn(Sys.Dir{1}) = 0;
    dirn = find(dirn);
    St = (Sys.St(dir, dir));
    Tt = (Sys.Tt(dir, dir));
    G = (Sys.G(dir, dirn)).';
    Sz = (Sys.S(dirn, dirn));
    Tz = (Sys.T(dirn, dirn));
    e = eigs(St, epsr * Tt, 2, (2 * pi * 5e9 / 3e8) ^ 2);
    e = e(abs(e) > 1e-5);
    val1 = sqrt(e(1)) * 3e8 / 2 / pi
end

%% Compare with analytical
m = 1;
n = 0;
fc = sqrt((m * pi / a) .^ 2 + (n * pi / b) .^ 2) * 3e11 / 2 / pi;
disp(abs(val0 - fc) / fc)
disp(abs(val1 - fc) / fc)

%% Dispersion diagram calculation
nmodes = 4;
ak = linspace(3e9, 20e9, 51) * 2 * pi / 3e8;
g = [];
B = Tt;
C = G;
opts.disp = 0;

tic
for i = 1:length(ak)
    k = ak(i) * sqrt(epsr);
    A = St - k ^ 2 * Tt;
    D = Sz - k ^ 2 * Tz;
    Sf = sparse(blkdiag(A, zeros(Sys.NDOFs - length(Sys.Dir{1}))));
    Tf = sparse([B C'; C D]);
    [vec, e] = eigs(Tf \ Sf, nmodes, 'SR', opts);
    e = sort(diag(real(sqrt(-e))), 'descend');
    g(:, i) = e;
end
time = toc / length(ak)

%% Plot dispersion
figure
plot(ak * 3e8 / 2 * pi, g(1:nmodes, :).');
axis tight
