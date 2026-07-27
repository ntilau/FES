%% ProjectScatteringFloquet - Scattering from periodic structure with Floquet BCs
clear all;
Config();

%% Setup
db = @(x) 20 * log10(x);
functPlot = @(x) abs(x);
theta = 0;
Sys.pOrd = 3;
Sys.hOrd = 1;
Sys.k = 2 * pi;
Sys.z0 = 120 * pi;
Sys.kEinc = [cosd(theta) sind(theta) 0];

Mesh = IOrPoly('ModelScatteringSquareFlocket', 'q34a0.005A', Sys.hOrd, 1);
Mesh.BC.Dir = 1;
Mesh.BC.ABC = 13;
Mesh.BC.DD = 12;
Mesh.epsr = [1 1];
Mesh.mur = {eye(2) eye(2)};

[Sys, Mesh] = AssembLin(Sys, Mesh);

%% Build system with DD (quasi-periodic) BCs
Sys.A = [(Sys.S - Sys.k ^ 2 * Sys.T + 1i * Sys.k * Sys.ABC), ...
    Sys.DD(:, Sys.DirDD); ...
    Sys.DD(Sys.DirDD, :), 1i * Sys.k * Sys.DD(Sys.DirDD, Sys.DirDD)];
Sys.b = [1i * Sys.k * Sys.fs; zeros(length(Sys.DirDD), 1)];

condest(Sys.A)

if isfield(Sys, 'Dir')
    Sys.A(Sys.Dir{1}, :) = 0;
    Sys.A(:, Sys.Dir{1}) = 0;
    Sys.A(Sys.Dir{1}, Sys.Dir{1}) = eye(length(Sys.Dir{1}));
end

%% Solve
tic
Sys.u = full(Sys.A \ Sys.b);
fprintf('Direct solver: %g s\n', toc);

%% Postprocessing
Sys.u = Sys.u(1:Sys.NDOFs);
figure;
pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
    'xydata', abs(Sys.u), ...
    'zdata', abs(Sys.u), ...
    'mesh', 'off', ...
    'colormap', 'jet', 'xygrid', 'on');
axis equal;
axis tight;
camlight left;
lighting phong;
