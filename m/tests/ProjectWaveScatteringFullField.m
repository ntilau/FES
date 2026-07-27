%% ProjectWaveScatteringFullField - Solve full-field wave scattering (total field formulation)
clear all;
Config();

%% Setup
db = @(x) 20 * log10(x);
functPlot = @(x) abs(x);
theta = 0;
Sys.pOrd = 2;
Sys.hOrd = 1;
Sys.k = 2 * pi;
Sys.z0 = 120 * pi;
Sys.kEinc = [cosd(theta) sind(theta) 0];

Mesh = IOrPoly('ModelScatteringSquare', 'q34a0.001A', Sys.hOrd, 1);
Mesh.BC.Dir = 1;
Mesh.BC.ABC = 133;
Mesh.epsr = [1 1];

[Sys, Mesh] = AssembLin(Sys, Mesh);

%% Build system
Sys.A = (Sys.S - Sys.k ^ 2 * Sys.T + 1i * Sys.k * Sys.ABC);
Sys.b = 1i * Sys.k * Sys.fs;

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
IOwVTK(Sys, Mesh, 'prova')

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
