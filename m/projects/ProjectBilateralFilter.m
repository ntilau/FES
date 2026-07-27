%% ProjectBilateralFilter - Compute S-parameters of a bilateral fin-line filter
clear all;
close all
Config();

%% Setup
Sys.pOrd = 2;
Sys.hOrd = 1;
prjName = 'BilatFilter';

Mesh = IOrPoly(prjName, 'q34a1A', Sys.hOrd, 1e-6);
Sys.Height = 1.651e-3 / 2;
Sys.WPnModes = 15;
Sys.WPportPlot = 1;
Sys.WPmodePlot = 1;
Sys.WPpow = 1;
Mesh.epsr = [1 2.1];
Mesh.BC.Dir = 1;
Mesh.BC.WP = [11 12];

[Sys, Mesh] = AssembLin(Sys, Mesh);

%% Frequency loop
Sys.nFreqs = 81;
Sys.freqs = linspace(138e9, 158e9, Sys.nFreqs);
Sys.fPlot = Sys.freqs(1);
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
    if freq == Sys.fPlot
        Sys.uPlot = Sys.u;
    end
end

%% Postprocessing: S-parameters plot
figure;
plot(Sys.freqs * 1e-9, ...
    Sys.db(Sys.Sparams(Sys.WPmodePlot:Sys.WPnModes:end, :).'));
axis([138 158 -40 0])

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
