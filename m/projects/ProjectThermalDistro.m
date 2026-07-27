%% ProjectThermalDistro - Solve steady-state heat distribution problem
clear; close all;
Config();

%% Setup
db = @(x) 20 * log10(abs(x));
functPlot = @(x) abs(x);

Sys.pOrd = 1;
Sys.hOrd = 1;

Mesh = IOrPoly('ModelHeatEquationDD', 'q30a0.1A', Sys.hOrd, 1);
Mesh.BC.Dir = [2 3];
Mesh.BC.Neu = 3;
Sys.ht = 750;
Sys.K = 52;
Sys.Tedge{1} = 300;
Sys.Tedge{2} = 200;
Sys.Text = 273.15;
Sys.q0 = 100;

[Sys, Mesh] = AssembLin(Sys, Mesh);

%% Build linear system with Dirichlet BCs
Sys.A = Sys.K * Sys.S;
Sys.b = Sys.q0 * Sys.fs;

if isfield(Sys, 'Dir')
    for ibc = 1:length(Sys.Dir)
        Sys.b = Sys.b + Sys.A(:, Sys.Dir{ibc}) ...
            * ones(length(Sys.Dir{ibc}), 1) * Sys.Tedge{ibc};
    end
    for ibc = 1:length(Sys.Dir)
        Sys.b(Sys.Dir{ibc}) = Sys.Tedge{ibc};
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
figure;
pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
    'xydata', (functPlot(Sys.u)), ...
    'zdata', (functPlot(Sys.u)), ...
    'mesh', 'off', ...
    'colormap', 'hot', 'xygrid', 'on');
