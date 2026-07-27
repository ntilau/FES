%% ProjectWaveScatteringDDglob - Wave scattering with global DD system
clear; close all;
Config()

%% Setup
functPlot = @(x) abs(x);
theta = 0;
Sys.pOrd = 1;
Sys.hOrd = 1;
Sys.k = 2 * pi;
Sys.Z0 = 120 * pi;
Sys.kEinc = [cosd(theta) sind(theta) 0];
Sys.toll = 0.001;

Mesh = IOrPoly('ModelScatteringDD', 'q34a0.005A', Sys.hOrd, 1);
Mesh.BC.Dir = 1;
Mesh.BC.ABC = 133;
Mesh.epsr = [1 1];

%% Assemble system matrices
[Sys, Mesh] = AssembLin(Sys, Mesh);

Mesh.BC.DD = 13;
[Sys1, Mesh1] = AssembDD(Sys, Mesh, 1);
[Sys2, Mesh2] = AssembDD(Sys, Mesh, 2);

%% Assemble full system
Sys.A = (Sys.S - Sys.k ^ 2 * Sys.T + 1i * Sys.k * Sys.ABC);
Sys.B = (Sys.S - Sys.k ^ 2 * Sys.T);

if isfield(Sys, 'Dir')
    Sys.A(Sys.Dir{1}, :) = 0;
    Sys.A(:, Sys.Dir{1}) = 0;
    Sys.A(Sys.Dir{1}, Sys.Dir{1}) = eye(length(Sys.Dir{1}));
    Sys.b = Sys.B(:, Sys.Dir{1}) * Sys.fsEinc(Sys.Dir{1});
    Sys.b(Sys.Dir{1}) = -Sys.fsEinc(Sys.Dir{1});
    Sys.B(Sys.Dir{1}, :) = 0;
    Sys.B(:, Sys.Dir{1}) = 0;
    Sys.B(Sys.Dir{1}, Sys.Dir{1}) = eye(length(Sys.Dir{1}));
else
    Sys.b = -Sys.B * Sys.fsEinc + Sys.f;
end

%% Assemble subdomain 1
Sys1.Ddd = Sys1.DD(:, Sys1.DirDD);
Sys1.Tdd = 1i / Sys.k * Sys1.DD(Sys1.DirDD, Sys1.DirDD);
Sys1.A = [(Sys1.S - Sys1.k ^ 2 * Sys1.T + 1i * Sys1.k * Sys1.ABC), Sys1.Ddd; ...
    Sys1.Ddd.', Sys1.Tdd];
Sys1.B = blkdiag((Sys1.S - Sys1.k ^ 2 * Sys1.T), ...
    eye(length(Sys1.DirDD)));
Sys1.T12 = [zeros(Sys.NDOFs), ...
    zeros(Sys.NDOFs, length(Sys1.DirDD)); ...
    Sys1.Ddd.', -Sys1.Tdd];

if isfield(Sys1, 'Dir')
    Sys1.A(Sys1.Dir, :) = 0;
    Sys1.A(:, Sys1.Dir) = 0;
    Sys1.A(Sys1.Dir, Sys1.Dir) = eye(length(Sys1.Dir));
    Sys1.b = Sys1.B(:, Sys1.Dir) * Sys1.fEinc(Sys1.Dir);
    Sys1.b(Sys1.Dir) = -Sys1.fEinc(Sys1.Dir);
    Sys1.B(Sys1.Dir, :) = 0;
    Sys1.B(:, Sys1.Dir) = 0;
    Sys1.B(Sys1.Dir, Sys1.Dir) = eye(length(Sys1.Dir));
else
    Sys1.b = -Sys1.B * Sys1.fEinc + Sys1.f;
end
idx1 = [Sys1.DirReg; Sys.NDOFs + (1:length(Sys1.DirDD)).'];
Sys1.A = Sys1.A(idx1, idx1);
Sys1.b = Sys1.b(idx1, 1);

%% Assemble subdomain 2
Sys2.Ddd = Sys2.DD(:, Sys2.DirDD);
Sys2.Tdd = 1i / Sys.k * Sys2.DD(Sys2.DirDD, Sys2.DirDD);
Sys2.A = [(Sys2.S - Sys2.k ^ 2 * Sys2.T + 1i * Sys2.k * Sys2.ABC), Sys2.Ddd; ...
    Sys2.Ddd.', Sys2.Tdd];
Sys2.B = blkdiag((Sys2.S - Sys2.k ^ 2 * Sys2.T), ...
    eye(length(Sys2.DirDD)));
Sys2.T21 = [zeros(Sys.NDOFs), ...
    zeros(Sys.NDOFs, length(Sys2.DirDD)); ...
    Sys2.Ddd.', -Sys2.Tdd];

if isfield(Sys2, 'Dir')
    Sys2.A(Sys2.Dir, :) = 0;
    Sys2.A(:, Sys2.Dir) = 0;
    Sys2.A(Sys2.Dir, Sys2.Dir) = eye(length(Sys2.Dir));
    Sys2.b = Sys2.B(:, Sys2.Dir) * Sys2.fEinc(Sys2.Dir);
    Sys2.b(Sys2.Dir) = -Sys2.fEinc(Sys2.Dir);
    Sys2.B(Sys2.Dir, :) = 0;
    Sys2.B(:, Sys2.Dir) = 0;
    Sys2.B(Sys2.Dir, Sys2.Dir) = eye(length(Sys2.Dir));
else
    Sys2.b = -Sys2.B * Sys2.fEinc + Sys2.f;
end
idx2 = [Sys2.DirReg; Sys.NDOFs + (1:length(Sys2.DirDD)).'];
Sys2.A = Sys2.A(idx2, idx2);
Sys2.b = Sys2.b(idx2, 1);

%% Solve full system (reference)
tic
Sys.u = Sys.A \ Sys.b;
fprintf('Direct solver: %g s\n', toc);
Sys.u = Sys.fsEinc + Sys.u(1:Sys.NDOFs);

%% DD iteration setup
Sys1.uf = zeros(Sys.NDOFs, 1);
Sys2.uf = zeros(Sys.NDOFs, 1);
Sys1.u = zeros(length(Sys1.DirReg) + length(Sys1.DirDD), 1);
Sys2.u = zeros(length(Sys2.DirReg) + length(Sys1.DirDD), 1);
Sys1.uft = zeros(Sys.NDOFs, 1);
Sys2.uft = zeros(Sys.NDOFs, 1);

Sys1.T12 = Sys1.T12(idx1, idx2);
Sys2.T21 = Sys2.T21(idx2, idx1);

%% Analyze global DD system eigenvalues
A = [(Sys1.S - Sys1.k ^ 2 * Sys1.T + 1i * Sys1.k * Sys1.ABC), -Sys1.T12; ...
    -Sys2.T21, (Sys2.S - Sys2.k ^ 2 * Sys2.T + 1i * Sys2.k * Sys2.ABC)];
P = [Sys1.A, 0 * Sys1.T12; 0 * Sys2.T21, Sys2.A];
e = eig(P \ full(A));
figure(1)
plot(e, '.')
axis equal

%% Domain decomposition iteration (alternating Schwarz)
error = 1;
i = 1;
while error > Sys.toll
    Sys1.u = Sys1.A \ (Sys1.b + Sys1.T12 * Sys2.u);
    Sys2.u = Sys2.A \ (Sys2.b + Sys2.T21 * Sys1.u);
    Sys1.ufp = Sys1.uf;
    Sys2.ufp = Sys2.uf;
    Sys1.uf(Sys1.DirReg, 1) = Sys1.u(1:length(Sys1.DirReg));
    Sys2.uf(Sys2.DirReg, 1) = Sys2.u(1:length(Sys2.DirReg));
    error1 = norm(Sys1.uf - Sys1.ufp) / norm(Sys1.uf);
    error2 = norm(Sys2.uf - Sys2.ufp) / norm(Sys2.uf);
    error(i) = max(error1, error2);
    fprintf('%g\n', error(i));
    i = i + 1;
end
fprintf('No of iterations = %d\n', i - 1);

%% Recover total field
Sys1.uf(Sys1.DirReg, 1) = Sys.fsEinc(Sys1.DirReg) ...
    + Sys1.u(1:length(Sys1.DirReg));
Sys2.uf(Sys2.DirReg, 1) = Sys.fsEinc(Sys2.DirReg) ...
    + Sys2.u(1:length(Sys2.DirReg));

Sys1.uf(Sys1.DirDD) = 0;
uDD = Sys1.uf + Sys2.uf;
Sys.u = uDD;

%% Postprocessing: field visualization
if exist('pdeplot', 'file')
    figure;
    pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
        'xydata', (abs(Sys.u)), ...
        'mesh', 'off', ...
        'colormap', 'jet', 'xygrid', 'off');
    axis equal;
    axis tight;
    camlight left;
    lighting phong;
else
    IOwVTK(Sys, Mesh, prjName);
end
