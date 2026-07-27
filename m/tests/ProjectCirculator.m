%% ProjectCirculator - Full-wave simulation of a ferrite circulator (S-parameters + H-field)
clear all;
close all
Config();

%% Setup
Sys.pOrd = 4;
Sys.hOrd = 1;
prjName = 'CircKoshiba26';

Mesh = IOrPoly(prjName, 'q34aAQ', Sys.hOrd, 1e-3);
Mesh.a = 22.86e-3;
Mesh.b = Mesh.a / 2;
Sys.WPnModes = 5;
Sys.WPportPlot = 1;
Sys.WPmodePlot = 1;
Sys.WPpow = 1;
Sys.Height = Mesh.b;
Mesh.epsr = [1 11.7];
Mesh.BC.Dir = 1;
Mesh.BC.WP = [11 12 13];

%% Ferrite material parameters
Ferr.Gamma = 1.759e7;  % [C/kg]
Ferr.Ms = 1317;        % Oe
Ferr.H0 = 200;         % G
Ferr.dH = 135;         % Oe*s
Ferr.w0 = Ferr.Gamma * Ferr.H0;
Ferr.wm = Ferr.Gamma * Ferr.Ms;
Ferr.aDH = Ferr.Gamma * Ferr.dH / 2;

%% Frequency loop
Sys.nFreqs = 1;
Sys.freqs = 9e9;
Sys.freqPlot = Sys.freqs;
Sys.Sparams = zeros(length(Mesh.BC.WP) * Sys.WPnModes, Sys.nFreqs);

for kf = 1:Sys.nFreqs
    freq = Sys.freqs(kf);
    fprintf('freq = %g GHz\n', freq / 1e9);
    omega = 2 * pi * freq;

    %% Update gyrotropic permeability
    mur = 1 + (Ferr.w0 + 1i * Ferr.aDH) * Ferr.wm ...
        ./ ((Ferr.w0 + 1i * Ferr.aDH) .^ 2 - omega .^ 2);
    kr = omega * Ferr.wm ...
        ./ ((Ferr.w0 + 1i * Ferr.aDH) .^ 2 - omega .^ 2);
    Mesh.mur = {eye(2), [mur 1i * kr; -1i * kr mur]};

    [Sys, Mesh] = AssembLin(Sys, Mesh);
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

    %% Recover field
    Sys.u = zeros(Sys.NDOFs, ...
        (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    Sys.u(Sys.nnWP) = X(length(Sys.WP) * Sys.WPnModes + 1:end, ...
        (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    for ip = 1:length(Sys.WP)
        Sys.u(Sys.WP{ip}) = Sys.WPgvec{ip} ...
            * X((1:Sys.WPnModes) + (ip - 1) * Sys.WPnModes, ...
            (Sys.WPportPlot - 1) * Sys.WPnModes + Sys.WPmodePlot);
    end
    if freq == Sys.freqPlot
        Sys.uPlot = Sys.u;
    end
end

%% Postprocessing: S-parameters
if Sys.nFreqs == 1
    Sys.db(Sys.Sparams(Sys.WPmodePlot:Sys.WPnModes:end, :))
else
    figure;
    plot(Sys.freqs, ...
        Sys.db(Sys.Sparams(Sys.WPmodePlot:Sys.WPnModes:end, :).'));
end

%% Postprocessing: field visualization
Sys.u = Sys.uPlot;
if exist('pdeplot', 'file')
    figure;
    pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
        'xydata', (abs(Sys.u)), ...
        'mesh', 'off', 'contour', 'on', 'levels', 10, ...
        'colormap', 'jet', 'xygrid', 'off');
    axis equal;
    axis tight;
    camlight left;
    lighting phong;
else
    IOwVTK(Sys, Mesh, prjName);
end

%% Magnetic field recovery
[Shape1, Shape1Deriv] = CalcShapeFunctions(1, Sys.pOrd);
[Shape2, Shape2DerivX, Shape2DerivY] = CalcShapeFunctions(2, Sys.pOrd);
Hconst = 1 / (1i * 2 * pi * freq * Sys.mu0);
Sys.u = zeros(Mesh.NELE, 2);
for ie = 1:Mesh.NELE
    [gIs] = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
    [detJ, invJt] = CalcJacobian(Mesh.node(Mesh.ele(ie, :), :));
    sol = Sys.uPlot(gIs);
    dE = invJt * [Shape2DerivX(0.5, 0.5); Shape2DerivY(0.5, 0.5)] * sol;
    Sys.u(ie, :) = Hconst ...
        * (Mesh.mur{Mesh.elab(ie)} \ [dE(2); -dE(1)]).';
end
disp(abs(Sys.u(9, :)))
IOwVTKH(Sys, Mesh, [prjName, 'H']);
