function Sys = AssembWPHBDDschur(Sys)
% Assemble harmonic-balance WP system with DD Schur-complement reordering
% of DoFs into interior-region and boundary blocks.

%% Remove Dirichlet and WP DoFs from the interior set
if ~isfield(Sys, 'nnWP')
    Sys.nnWP = 1:Sys.NDOFs * Sys.nHarms;
    remId    = [];
    for ibc = 1:length(Sys.Dir)
        remId = [remId; Sys.Dir{ibc}];
    end
    for ibc = 1:length(Sys.WP)
        for jh = 1:Sys.nHarms
            remId = [remId; Sys.WP{ibc} + Sys.NDOFs * (jh - 1)];
        end
    end
    Sys.nnWP(remId) = [];
end

k0      = 2 * pi * Sys.freq / Sys.c0;
nRHS    = length(Sys.WP) * Sys.WPnModes * Sys.nHarms;
Sys.A   = Sys.S - k0^2 * Sys.T;
Sys.B   = sparse(length(Sys.nnWP) + nRHS, nRHS);
PP_size = length(Sys.WP) * Sys.WPnModes * Sys.nHarms;
PP      = zeros(PP_size, PP_size);
IP      = zeros(length(Sys.nnWP), PP_size);

%% Loop over ports and harmonics
for ip = 1:length(Sys.WP)
    for jh = 1:Sys.nHarms
        if isfield(Sys, 'Einc')
            Sys.WPpowEq = diag((Sys.Einc).^2 * Sys.Height ...
                * (sqrt(1 - (Sys.WPfc{ip} ...
                  ./ (Sys.Harms(jh) * Sys.freq)).^2)) / (Sys.z0));
        else
            Sys.WPpowEq = Sys.WPpow * eye(Sys.WPnModes) * 2 / Sys.Height;
        end

        gamma = diag(1i * 2 * pi * (Sys.Harms(jh) * Sys.freq) / Sys.c0 ...
            * sqrt(1 - (Sys.WPfc{ip}.' ...
                  / (Sys.Harms(jh) * Sys.freq)).^2));
        gamma = abs(real(gamma)) + 1i * imag(gamma);

        Sys.WPgvec{ip, jh} = Sys.WPvec{ip} ...
            * (sqrt(1i * (Sys.Harms(jh) * k0) * Sys.z0 ...
                  * (gamma \ Sys.WPpowEq)));

        ppIdx = (1:Sys.WPnModes) ...
              + ((jh - 1) + (ip - 1) * Sys.nHarms) * Sys.WPnModes;

        PP(ppIdx, ppIdx) = ...
            Sys.WPgvec{ip, jh}.' ...
            * Sys.A(Sys.WP{ip} + Sys.NDOFs * (jh - 1), ...
                    Sys.WP{ip} + Sys.NDOFs * (jh - 1)) ...
            * Sys.WPgvec{ip, jh} ...
            + 1i * Sys.Harms(jh) * k0 * Sys.z0 * Sys.WPpowEq * eye(Sys.WPnModes);

        IP(:, ppIdx) = ...
            Sys.A(Sys.nnWP, Sys.WP{ip} + Sys.NDOFs * (jh - 1)) ...
            * Sys.WPgvec{ip, jh};

        Sys.B(ppIdx, ppIdx) = ...
            1i * Sys.Harms(jh) * k0 * Sys.z0 * 2 * Sys.WPpowEq * eye(Sys.WPnModes);
    end
end

PI = IP.';
II = Sys.A(Sys.nnWP, Sys.nnWP);
Sys.A = [PP, PI; IP, II];

SizScat = length(Sys.WP) * Sys.WPnModes * Sys.nHarms;

%% Apply scaling for fundamental and intermodulation tones
Sys.B(1, 1)              = Sys.B(1, 1) * Sys.Pfund;
Sys.A(1, 1)              = Sys.A(1, 1) * Sys.Pfund;
Sys.A(SizScat + 1:end, 1) = Sys.A(SizScat + 1:end, 1) * sqrt(Sys.Pfund);
Sys.A(1, SizScat + 1:end) = Sys.A(1, SizScat + 1:end) * sqrt(Sys.Pfund);

Sys.B(Sys.WPnum, 1)                             = Sys.B(Sys.WPnum, Sys.WPnum) * Sys.Pitrf;
Sys.A(Sys.WPnum, Sys.WPnum)                     = Sys.A(Sys.WPnum, Sys.WPnum) * Sys.Pitrf;
Sys.A(SizScat + 1:end, Sys.WPnum)               = Sys.A(SizScat + 1:end, Sys.WPnum) * sqrt(Sys.Pitrf);
Sys.A(Sys.WPnum, SizScat + 1:end)               = Sys.A(Sys.WPnum, SizScat + 1:end) * sqrt(Sys.Pitrf);

%% Reorder into Schur-complement blocks
Sys.RegDoFRed = cell(length(Sys.RegDoF), 1);
Sys.AII       = cell(length(Sys.RegDoF), 1);
Sys.AIF       = cell(length(Sys.RegDoF), 1);
Sys.AFI       = cell(length(Sys.RegDoF), 1);

roof = length(PP);
Sys.BndDoFRed = roof + (1:length(intersect(Sys.DDmap(Sys.BndDoF), Sys.nnWP)));
roof = roof + length(Sys.BndDoFRed);

idx = [];
for jh = 1:Sys.nHarms
    idx = [idx, Sys.BndDoFRed + (jh - 1) * length(Sys.nnWP) / Sys.nHarms];
end
Sys.BndDoFRed = [1:length(PP), idx];
Sys.AFF       = Sys.A(Sys.BndDoFRed, Sys.BndDoFRed);

for ir = 1:length(Sys.RegDoF)
    Sys.RegDoFRed{ir} = roof + (1:length(intersect(Sys.RegDoFmap{ir}, Sys.nnWP)));
    roof = roof + length(Sys.RegDoFRed{ir});
    idx = [];
    for jh = 1:Sys.nHarms
        idx = [idx, Sys.RegDoFRed{ir} + (jh - 1) * length(Sys.nnWP) / Sys.nHarms];
    end
    Sys.RegDoFRed{ir} = idx;
    Sys.AII{ir} = Sys.A(Sys.RegDoFRed{ir}, Sys.RegDoFRed{ir});
    Sys.AIF{ir} = Sys.A(Sys.RegDoFRed{ir}, Sys.BndDoFRed);
    Sys.AFI{ir} = Sys.A(Sys.BndDoFRed, Sys.RegDoFRed{ir});
end

Sys.WPGlobDoF = (1:length(Sys.WP) * Sys.WPnModes * Sys.nHarms).';
Sys.gF        = Sys.B(Sys.BndDoFRed, Sys.WPGlobDoF);

end
