%% ProjectCirculatorIMP - Circulator intermodulation analysis (HB + ferrite)
clear all;
Config();
format short

%% Setup
Sys.pOrd = 2;
Sys.hOrd = 1;
prjName = 'CircKoshiba26_5';

Mesh = IOrPoly(prjName, 'q34aAQ', Sys.hOrd, 1e-3);
Sys.WPnModes = 1;
Sys.WPportPlot = 1;
Sys.WPmodePlot = 1;
Sys.Height = 22.86e-03 / 2;
Sys.WPpow = 1;
Mesh.mur = [1 1];
Mesh.kr = [0 0];
Mesh.epsr = [1 1 1 1 11.7];
Mesh.BC.Dir = 1;
Mesh.BC.WP = [11 12 13];
Mesh.NLlab = 2;

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
Sys.HBharms = [f1 f2 2 * f1 - f2 2 * f2 - f1];
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

Mesh.MurMat = zeros(Sys.nHarms * Sys.nHarms, Mesh.NELE);
Mesh.KrMat = zeros(Sys.nHarms * Sys.nHarms, Mesh.NELE);
idxNL = find(Mesh.elab == Mesh.NLlab);

Sys = CalcDoFsNumber(Sys, Mesh);
Sys.u = zeros(Sys.NDOFs * Sys.nHarms, 1);
Sys.Pfund = 1;
Sys.Pitrf = 1;
Sys.WPnum = 6;

%% Frequency loop with nonlinear iteration
Sys.nFreqs = 11;
Sys.freqs = linspace(10e9, 12e9, Sys.nFreqs);
Sys.Sparams = zeros(...
    length(Mesh.BC.WP) * Sys.WPnModes * Sys.nHarms, Sys.nFreqs);

for kf = 1:Sys.nFreqs
    Sys.freq = Sys.freqs(kf);
    f1 = Sys.freq;
    f2 = 1.1e10;
    if f1 == f2
        continue;
    end
    Sys.HBharms = [f1 f2 2 * f1 - f2 2 * f2 - f1] / f1;

    fprintf('freq = %g GHz\n', Sys.freq / 1e9);
    error = 1;
    Sys.u0 = Sys.u;

    if ~isfield(Sys, 'SinOnly')
        omega(1:2:2 * length(Sys.HBharms), 1) = 2 * pi * Sys.freq * Sys.HBharms.';
        omega(2:2:2 * length(Sys.HBharms), 1) = 2 * pi * Sys.freq * Sys.HBharms.';
    else
        omega = 2 * pi * Sys.freq * Sys.HBharms.';
    end

    %% Update frequency-dependent permeability at each harmonic
    mur = diag(1 + ((Ferr.w0 * ones(Sys.nHarms, 1)) + 1i * Ferr.aDH) ...
        .* (Ferr.wm * ones(Sys.nHarms, 1)) ...
        ./ ((Ferr.w0 * ones(Sys.nHarms, 1) + 1i * Ferr.aDH) .^ 2 - (omega) .^ 2));
    kr = diag(omega ...
        .* (Ferr.wm * ones(Sys.nHarms, 1)) ...
        ./ ((Ferr.w0 * ones(Sys.nHarms, 1) + 1i * Ferr.aDH) .^ 2 - (omega) .^ 2));
    Mesh.MurMat(:, idxNL) = mur(:) * ones(1, length(idxNL));
    Mesh.KrMat(:, idxNL) = kr(:) * ones(1, length(idxNL));

    %% Nonlinear iteration
    while error > 1e-9
        [Sys, Mesh] = AssembHBFerrite(Sys, Mesh);
        Sys = AssembWPHB(Sys);
        X = Sys.A \ Sys.B;

        %% Extract S-parameters
        sp = X(1:length(Sys.WP) * Sys.WPnModes * Sys.nHarms, 1);
        sp(1, 1) = (sp(1, 1) - 1) * sqrt(Sys.Pfund);
        sp(Sys.WPnum, 1) = (sp(Sys.WPnum, 1) - 1) * sqrt(Sys.Pitrf);
        Sys.Sparams(:, kf) = sp(:, 1);

        %% Recover field
        Sys.u = zeros(Sys.NDOFs * Sys.nHarms, 1);
        Sys.u(Sys.nnWP) = X(...
            length(Sys.WP) * Sys.WPnModes * Sys.nHarms + 1:end, 1);
        for jh = 1:length(Sys.Harms)
            for ip = 1:length(Sys.WP)
                Sys.u((jh - 1) * Sys.NDOFs + Sys.WP{ip}) ...
                    = Sys.WPgvec{ip, jh} ...
                    * X((1:Sys.WPnModes) ...
                    + ((jh - 1) + (ip - 1) * Sys.nHarms) * Sys.WPnModes, 1);
            end
        end

        error = norm(Sys.u - Sys.u0) / norm(Sys.u);
        Sys.u0 = Sys.u;
        fprintf('%2.6g\n', error);
    end
    if Sys.freq == 10e9
        Sys.uPlot = Sys.u;
    end
end

%% Remove degenerate frequency points
idx = find(Sys.freqs == f2);
Sys.freqs(idx) = [];
Sys.Sparams(:, idx) = [];

%% Postprocessing: S-parameters
if Sys.nFreqs == 1
    PowDist = Sys.db(Sys.Sparams(Sys.WPmodePlot:Sys.WPnModes:end, 1))
else
    figure;
    plot(Sys.freqs, ...
        Sys.db(Sys.Sparams(Sys.WPmodePlot:Sys.WPnModes:end, :).'));
end

%% Postprocessing: field per harmonic
for ih = 1:Sys.HBharmPlot
    Sys.u = Sys.uPlot((ih - 1) * Sys.NDOFs + (1:Sys.NDOFs));
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
