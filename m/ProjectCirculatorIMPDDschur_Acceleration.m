%% ProjectCirculatorIMPDDschur_Acceleration - Accelerated circulator IMP with DD Schur complement
clear all; clc;
close all
Config();
format short
profile -memory on

%% Setup
model = 1;
HarmsRetained = 4;
Sys.pOrd = 1;
Sys.hOrd = 1;

switch model
    case 1
        prjName = 'CircKoshiba26';
    case 2
        prjName = 'CircKoshiba26_5';
end
Mesh = IOrPoly(prjName, 'q34aAQ', Sys.hOrd, 1e-3);

Sys.WPnModes = 1;
Sys.WPportPlot = 1;
Sys.WPmodePlot = 1;
Sys.Height = 22.86e-03 / 2;
Sys.WPpow = 1;
Mesh.mur = [1 1];
Mesh.kr = [0 0];

switch model
    case 1
        Mesh.epsr = [1 11.7];
    case 2
        Mesh.epsr = [1 1 1 1 11.7];
end
Mesh.BC.Dir = 1;
Mesh.BC.WP = [11 12 13];
Mesh.NLlab = length(Mesh.epsr);

%% Ferrite material parameters
Ferr.Gamma = 1.759e7;  % [C/kg]
Ferr.Ms = 1317;        % Oe
Ferr.H0 = 200;         % G
Ferr.dH = 135;         % Oe*s
Ferr.w0 = Ferr.Gamma * Ferr.H0;
Ferr.wm = Ferr.Gamma * Ferr.Ms;
Ferr.aDH = Ferr.Gamma * Ferr.dH / 2;
Ferr.alpha = 1;
Mesh.Ferr = Ferr;

%% Harmonic balance setup
f1 = 1;
f2 = 1.1;
Sys.HBharms = [f1 f2 2 * f1 - f2 2 * f2 - f1, ...
    3 * f1 3 * f2 2 * f1 + f2 2 * f2 + f1];
Sys.HBharms = Sys.HBharms(1:HarmsRetained);
Sys.HBharmPlot = length(Sys.HBharms);
Sys.OverSampling = 2;
Sys.SinOnly = true;
Sys.nHarms = length(Sys.HBharms);

if ~isfield(Sys, 'SinOnly')
    Sys.nHarms = length(Sys.HBharms) * 2;
else
    if ~Sys.SinOnly
        Sys.HBharmPlot = 2 * Sys.HBharmPlot - 1;
    end
end

%% Domain decomposition setup
Mesh.BC.DDschur = 2;
[Mesh, Sys] = GetBndMap(Sys, Mesh);

Mesh.MurMat = zeros(Sys.nHarms * Sys.nHarms, Mesh.NELE);
Mesh.KrMat = zeros(Sys.nHarms * Sys.nHarms, Mesh.NELE);
idxNL = find(Mesh.elab == Mesh.NLlab);

Sys = CalcDoFsNumber(Sys, Mesh);
Sys.u = zeros(Sys.NDOFs * Sys.nHarms, 1);

Sys.Pfund = 150;
Sys.Pitrf = 1500;
Sys.WPnum = 2 * Sys.WPnModes * Sys.nHarms;
Sys.Power = ones(1, length(Mesh.BC.WP) * Sys.WPnModes * Sys.nHarms);
Sys.Power(1) = sqrt(Sys.Pfund);
Sys.Power(Sys.WPnum) = sqrt(Sys.Pitrf);

%% Frequency loop
Sys.Compute = ones(1, length(Sys.RegDoF));
Sys.nFreqs = 1;
Sys.freqs = 9.1e9;
Sys.Sparams = zeros(...
    length(Mesh.BC.WP) * Sys.WPnModes * Sys.nHarms, Sys.nFreqs);

for kf = 1:Sys.nFreqs
    Sys.freq = Sys.freqs(kf);
    f1 = Sys.freq;
    f2 = 1e10;
    if f1 == f2
        continue;
    end
    Sys.HBharms = [f1 f2 2 * f1 - f2 2 * f2 - f1, ...
        3 * f1 3 * f2 2 * f1 + f2 2 * f2 + f1] / f1;
    Sys.HBharms = Sys.HBharms(1:HarmsRetained);

    fprintf('freq = %g GHz\n', Sys.freq / 1e9);
    errorD = 1;
    Sys.u0 = Sys.u;

    if ~isfield(Sys, 'SinOnly')
        omega(1:2:2 * length(Sys.HBharms), 1) ...
            = 2 * pi * Sys.freq * Sys.HBharms.';
        omega(2:2:2 * length(Sys.HBharms), 1) ...
            = 2 * pi * Sys.freq * Sys.HBharms.';
    else
        omega = 2 * pi * Sys.freq * Sys.HBharms.';
    end

    %% Frequency-dependent permeability at each harmonic
    mur = diag(1 + ((Ferr.w0 * ones(Sys.nHarms, 1)) + 1i * Ferr.aDH) ...
        .* (Ferr.wm * ones(Sys.nHarms, 1)) ...
        ./ ((Ferr.w0 * ones(Sys.nHarms, 1) + 1i * Ferr.aDH) .^ 2 - (omega) .^ 2));
    kr = diag(omega ...
        .* (Ferr.wm * ones(Sys.nHarms, 1)) ...
        ./ ((Ferr.w0 * ones(Sys.nHarms, 1) + 1i * Ferr.aDH) .^ 2 - (omega) .^ 2));
    Mesh.MurMat(:, idxNL) = mur(:) * ones(1, length(idxNL));
    Mesh.KrMat(:, idxNL) = kr(:) * ones(1, length(idxNL));

    %% Copy for full solve
    SysF = Sys;
    MeshF = Mesh;

    tD = 0;
    tF = 0;

    %% Nonlinear iteration
    while errorD > 1e-9
        % Full solve
        ttF = tic;
        [SysF, MeshF, errorF, spF] = AssembSolveFull(SysF, MeshF);
        tF = tF + toc(ttF);

        % DD solve
        ttD = tic;
        [Sys, Mesh, errorD, spD] = AssembSolveDD(Sys, Mesh);
        tD = tD + toc(ttD);

        Sys.Sparams(:, kf) = spF(:, 1);
        fprintf('%2.6e\t\t%2.6e\n', errorD, errorF);
    end
    if Sys.freq == 9e9
        Sys.uPlot = SysF.u;
    end
end

%% Report acceleration and statistics
Acc = tF / tD;
fprintf('DoFs = %g, NNZ = %d, Acc = %2.4g, p = %d, tF = %g, tD = %g\n', ...
    length(Sys.A), nnz(Sys.A), Acc, Sys.pOrd, tF, tD);
profile viewer

%% Postprocessing: field per harmonic
for ih = 1:Sys.HBharmPlot
    Sys.u = Sys.uPlot((ih - 1) * SysF.NDOFs + (1:SysF.NDOFs));
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
end
