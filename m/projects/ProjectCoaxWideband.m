%% PrjCoaxWideband - Compute capacitance of a coaxial cable
clear all;
Config();

%% Setup
functPlot = @(x) abs(x);

Sys.pOrd = 4;
Sys.hOrd = 1;
prjName = 'coax';

% type | impedance ohms | core mm | Dielectric Type | Dielectric VF
% RG-58/U | 50 | 0.81 | PE | 0.66
% Dielectric mm | OD mm | shields | max attenuation @ 750 MHz dB/100 ft
% 2.9 | 5.0 | single | 13.104
WriteCoax(0.8e-3 / 2, 2.95e-3 / 2, 0.01e-3);
scale = 1;
Mesh = IOrPoly(prjName, 'q34aA', Sys.hOrd, scale);

Mesh.epsr = [1 (1 / 0.66) ^ 2];
Mesh.BC.Dir = [1 2];
Sys.V{1} = 1;
Sys.V{2} = 0;

[Sys, Mesh] = AssembLinStatic(Sys, Mesh);

%% Build linear system with Dirichlet BCs
Sys.A = Sys.S;
Sys.b = 0 * Sys.fs;

if isfield(Sys, 'Dir')
    for ibc = 1:length(Sys.Dir)
        Sys.b = Sys.b - Sys.A(:, Sys.Dir{ibc}) ...
            * ones(length(Sys.Dir{ibc}), 1) * Sys.V{ibc};
    end
    for ibc = 1:length(Sys.Dir)
        Sys.b(Sys.Dir{ibc}) = Sys.V{ibc};
        Sys.A(Sys.Dir{ibc}, :) = 0;
        Sys.A(:, Sys.Dir{ibc}) = 0;
        Sys.A(Sys.Dir{ibc}, Sys.Dir{ibc}) = eye(length(Sys.Dir{ibc}));
    end
end

%% Solve and compute capacitance
tic
Sys.u = Sys.A \ Sys.b;
fprintf('Direct solver: %g s\n', toc);

eps0 = 8.854187817;
W = 0.5 * eps0 * Sys.u.' * Sys.S * Sys.u;
Cap = 2 * W;
fprintf('Capacitance = %g pF/m\n', Cap)

%% Postprocessing
if exist('pdeplot', 'file')
    PlotPoly(prjName, figure, scale);
    hold on;
    pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
        'xydata', (functPlot(Sys.u)), ...
        'mesh', 'off', ...
        'colormap', 'jet', 'xygrid', 'off');
    axis equal;
    camlight left;
    lighting phong;
else
    IOwVTK(Sys, Mesh, prjName);
end
