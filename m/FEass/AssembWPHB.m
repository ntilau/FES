function Sys = AssembWPHB(Sys)
% Assemble harmonic-balance waveguide-port augmented system matrix A
% with mode expansion at each port and each harmonic frequency.

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

%% Assemble augmented system
PI = IP.';
II = Sys.A(Sys.nnWP, Sys.nnWP);
Sys.A = [PP, PI; IP, II];

end
