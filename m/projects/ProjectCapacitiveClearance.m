%% ProjectCapacitiveClearance - Compute capacitance of a capacitive sensor
clear all;
Config();

%% Setup
functPlot = @(x) abs(x);

Sys.pOrd = 4;
Sys.hOrd = 1;
prjName = 'CapSense';
scale = 1e-3;

Mesh = IOrPoly(prjName, 'q34aA', Sys.hOrd, scale);
Mesh.epsr = [1 10];
Mesh.BC.Dir = [1 2];
Sys.V{1} = 0;
Sys.V{2} = 1;

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

Cap = 0.5 * Sys.u.' * Sys.S * Sys.u;
fprintf('Capacitance = %g F/m\n', Cap)

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
    IOwVTK(Sys, Mesh, 'volt');
end
