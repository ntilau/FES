function Sys = AssembWPDDschur(Sys, freq)
% Assemble the waveguide-port system matrix with DD Schur-complement
% reordering of DoFs into interior-region and boundary blocks.

%% Remove Dirichlet and WP DoFs from the interior set
if ~isfield(Sys, 'nnWP')
    Sys.nnWP  = 1:Sys.NDOFs;
    remId     = [];
    for ibc = 1:length(Sys.Dir)
        remId = [remId; Sys.Dir{ibc}];
    end
    for ibc = 1:length(Sys.WP)
        remId = [remId; Sys.WP{ibc}];
    end
    Sys.nnWP(remId)  = [];
    Sys.remDoF       = unique(remId);
end

k0 = 2 * pi * freq / Sys.c0;

nRHS     = length(Sys.WP) * Sys.WPnModes;
Sys.A    = Sys.S - k0^2 * Sys.T;
Sys.B    = sparse(length(Sys.nnWP) + nRHS, nRHS);
PP       = zeros(length(Sys.WP) * Sys.WPnModes, ...
                length(Sys.WP) * Sys.WPnModes);
IP       = zeros(length(Sys.nnWP), length(Sys.WP) * Sys.WPnModes);

%% Loop over waveguide ports
for ip = 1:length(Sys.WP)
    if isfield(Sys, 'Einc')
        Sys.WPpowEq = diag((Sys.Einc).^2 * Sys.Height ...
            * (sqrt(1 - (Sys.WPfc{ip} ./ freq).^2)) / (Sys.z0));
    else
        Sys.WPpowEq = Sys.WPpow * eye(Sys.WPnModes) * 2 / Sys.Height;
    end

    gamma = diag(1i * 2 * pi * freq / Sys.c0 ...
                 * sqrt(1 - (Sys.WPfc{ip}.' / freq).^2));
    gamma = abs(real(gamma)) + 1i * imag(gamma);

    Sys.WPgvec{ip} = Sys.WPvec{ip} ...
        * (sqrt(1i * k0 * Sys.z0 * (gamma \ Sys.WPpowEq)));

    ppIdx = (1:Sys.WPnModes) + (ip - 1) * Sys.WPnModes;
    PP(ppIdx, ppIdx) = ...
        Sys.WPgvec{ip}.' * Sys.A(Sys.WP{ip}, Sys.WP{ip}) ...
        * Sys.WPgvec{ip} + 1i * k0 * Sys.z0 * Sys.WPpowEq * eye(Sys.WPnModes);

    IP(:, ppIdx) = Sys.A(Sys.nnWP, Sys.WP{ip}) * Sys.WPgvec{ip};

    Sys.B(ppIdx, ppIdx) = ...
        1i * k0 * Sys.z0 * 2 * Sys.WPpowEq * eye(Sys.WPnModes);
end

PI = IP.';
II = Sys.A(Sys.nnWP, Sys.nnWP);
Sys.A = [PP, PI; IP, II];

%% Reorder into Schur-complement blocks
Sys.RegDoFRed = cell(length(Sys.RegDoF), 1);
Sys.AII       = cell(length(Sys.RegDoF), 1);
Sys.AIF       = cell(length(Sys.RegDoF), 1);
Sys.AFI       = cell(length(Sys.RegDoF), 1);
Sys.BndDoFRed = 1:length(Sys.BndDoF);

roof = length(Sys.BndDoFRed);
for ir = 1:length(Sys.RegDoF)
    Sys.RegDoFRed{ir} = roof + (1:length(intersect(Sys.RegDoFmap{ir}, Sys.nnWP)));
    roof = roof + length(Sys.RegDoFRed{ir});

    Sys.AII{ir} = Sys.A(Sys.RegDoFRed{ir}, Sys.RegDoFRed{ir});
    Sys.AIF{ir} = Sys.A(Sys.RegDoFRed{ir}, Sys.BndDoFRed);
    Sys.AFI{ir} = Sys.A(Sys.BndDoFRed, Sys.RegDoFRed{ir});
end

Sys.WPGlobDoF = (1:length(Sys.WP) * Sys.WPnModes).';
Sys.AFF       = Sys.A(Sys.BndDoFRed, Sys.BndDoFRed);
Sys.gF        = Sys.B(Sys.BndDoFRed, Sys.WPGlobDoF);

end
