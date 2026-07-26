%% ProjectCirculatorDDschur - Circulator with domain decomposition (Schur complement) and iterative solver
clear all;
close all
Config();

%% Setup
Sys.pOrd = 1;
Sys.hOrd = 1;
prjName = 'CircKoshiba26_5';

Mesh = IOrPoly(prjName, 'q34a1AQ', Sys.hOrd, 1e-3);
Mesh.a = 22.86e-3;
Mesh.b = Mesh.a / 2;
Sys.WPnModes = 1;
Sys.WPportPlot = 1;
Sys.WPmodePlot = 1;
Sys.WPpow = 1;
Sys.Height = Mesh.b;
Mesh.epsr = [1 1 1 1 11.7];
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

%% Domain decomposition setup
Mesh.NLlab = 5;
Mesh.BC.DDschur = 2;
[Mesh, Sys] = GetBndMap(Sys, Mesh);

%% Frequency loop
Sys.nFreqs = 1;
Sys.freqs = 10e9;
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
    Mesh.mur = {eye(2), eye(2), eye(2), eye(2), ...
        [mur 1i * kr; -1i * kr mur]};

    [Sys, Mesh] = AssembLinDDschur(Sys, Mesh);
    Sys = AssembWPDDschur(Sys, freq);

    tic
    X = SolvDir(Sys);
    toc

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

%% PCG iterative solver test
setup.type = 'crout';
setup.milu = 'row';
setup.droptol = 1e-6;
[L, U] = ilu(Sys.A, setup);
P = tril(Sys.A);
u0 = Sys.B(:, 1) * 0;
r0 = Sys.B(:, 1) - Sys.A * u0;
z0 = P \ r0;
p0 = z0;
err = 1;
i = 1;
errv = [];

while err > 1e-9
    a = r0.' * z0 / (p0.' * Sys.A * p0);
    u1 = u0 + a * p0;
    err = norm(u0 - u1) / norm(u0);
    errv(i) = err;
    fprintf('%d\n', err);
    i = i + 1;

    r1 = r0 - a * Sys.A * p0;
    z1 = P \ r1;
    b = z1.' * r1 / (z0.' * r0);
    p0 = z1 + b * p0;
    r0 = r1;
    z0 = z1;
    u0 = u1;
end

figure;
semilogy(errv)

%% Postprocessing: field visualization
Sys.u = Sys.uPlot(Sys.DDmap);
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
