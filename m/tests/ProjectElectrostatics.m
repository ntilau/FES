%% ProjectElectrostatics - Solve electrostatic potential on a scattering geometry
clear all;
close all;
Config();

%% Setup
functPlot = @(x) abs(x);

Sys.pOrd = 4;
Sys.hOrd = 1;

Mesh = IOrPoly('ModelScatteringDD', 'q34a0.01A', Sys.hOrd, 1);
Mesh.BC.Dir = [1 133];
Sys.V{1} = 1;
Sys.V{2} = 0;

[Sys, Mesh] = AssembLin(Sys, Mesh);

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

%% Solve
tic
Sys.u = Sys.A \ Sys.b;
fprintf('Direct solver: %g s\n', toc);

%% Postprocessing
IOwVTK(Sys, Mesh, 'volt');

figure;
pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
    'xydata', (functPlot(Sys.u)), ...
    'zdata', (functPlot(Sys.u)), ...
    'mesh', 'off', ...
    'colormap', 'jet', 'xygrid', 'off');
axis equal;
camlight left;
lighting phong;
