%% ProjectWaveGuide - Compute S-parameters and field distribution in a rectangular waveguide
clear all; clc;
close all
Config();

%% Setup
Sys.pOrd = 4;
Sys.hOrd = 1;
prjName = 'WaveGuide';

WriteWaveGuide(22.86, 22.86);
Mesh = IOrPoly(prjName, 'q34aA', Sys.hOrd, 1e-3);
Mesh.a = 22.86e-3;
Mesh.b = Mesh.a / 2;
Sys.Height = Mesh.b;
Sys.WPnModes = 1;
Sys.WPportPlot = 1;
Sys.WPmodePlot = 1;
Sys.WPpow = 1;
Mesh.BC.Dir = 1;
Mesh.BC.WP = [11 12];

[Sys, Mesh] = AssembLin(Sys, Mesh);

%% Frequency loop
Sys.nFreqs = 1;
Sys.freqs = 1e10;
Sys.Sparams = zeros(length(Mesh.BC.WP) * Sys.WPnModes, Sys.nFreqs);

for kf = 1:Sys.nFreqs
    freq = Sys.freqs(kf);
    fprintf('freq = %g GHz\n', freq / 1e9);

    Sys = AssembWP(Sys, freq);
    X = Sys.A \ Sys.B;

    %% Extract S-parameters
    sp = X(1:length(Sys.WP) * Sys.WPnModes, ...
        1:length(Sys.WP) * Sys.WPnModes) ...
        - eye(length(Sys.WP) * Sys.WPnModes);
    Sys.Sparams(:, kf) = sp(:, ...
        (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    fprintf('  losses = %2.2g%%\n', ...
        (1 - norm(Sys.Sparams(:, kf))) * 100);

    %% Recover field solution
    Sys.u = zeros(Sys.NDOFs, ...
        (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    Sys.u(Sys.nnWP) = X(length(Sys.WP) * Sys.WPnModes + 1:end, ...
        (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    for ip = 1:length(Sys.WP)
        Sys.u(Sys.WP{ip}) = Sys.WPgvec{ip} ...
            * X((1:Sys.WPnModes) + (ip - 1) * Sys.WPnModes, ...
            (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    end
    if freq == 10e9
        Sys.uPlot = Sys.u;
    end
end

%% Postprocessing: field visualization
Sys.u = Sys.uPlot;
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

max(abs(Sys.u))
