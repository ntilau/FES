function Sys = AssembWP(Sys, freq)
% Assemble the waveguide-port (WP) augmented system matrix A
% by adding absorbing port boundary conditions (PML-like) to the
% interior problem at a given frequency.

%% Remove Dirichlet and WP DoFs from the interior set
if ~isfield(Sys, 'nnWP')
    Sys.nnWP = 1:Sys.NDOFs;
    remId    = [];
    for ibc = 1:length(Sys.Dir)
        remId = [remId; Sys.Dir{ibc}];
    end
    for ibc = 1:length(Sys.WP)
        remId = [remId; Sys.WP{ibc}];
    end
    Sys.nnWP(remId) = [];
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
    %% Port power equalization
    if isfield(Sys, 'Einc')
        Sys.WPpowEq = diag((Sys.Einc).^2 * Sys.Height ...
            * (sqrt(1 - (Sys.WPfc{ip} ./ freq).^2)) / (Sys.z0));
    else
        Sys.WPpowEq = Sys.WPpow * eye(Sys.WPnModes) * 2 / Sys.Height;
    end

    %% Propagation constant
    gamma = diag(1i * 2 * pi * freq / Sys.c0 ...
                 * sqrt(1 - (Sys.WPfc{ip}.' / freq).^2));
    gamma = abs(real(gamma)) + 1i * imag(gamma);

    %% Port mode coupling
    Sys.WPgvec{ip} = Sys.WPvec{ip} ...
        * (sqrt(1i * k0 * Sys.z0 * (gamma \ Sys.WPpowEq)));

    %% Port-port block
    ppIdx = (1:Sys.WPnModes) + (ip - 1) * Sys.WPnModes;
    PP(ppIdx, ppIdx) = ...
        Sys.WPgvec{ip}.' * Sys.A(Sys.WP{ip}, Sys.WP{ip}) ...
        * Sys.WPgvec{ip} + 1i * k0 * Sys.z0 * Sys.WPpowEq * eye(Sys.WPnModes);

    %% Interior-port coupling block
    IP(:, ppIdx) = Sys.A(Sys.nnWP, Sys.WP{ip}) * Sys.WPgvec{ip};

    %% Damping matrix block
    Sys.B(ppIdx, ppIdx) = ...
        1i * k0 * Sys.z0 * 2 * Sys.WPpowEq * eye(Sys.WPnModes);
end

%% Assemble augmented system
PI = IP.';
II = Sys.A(Sys.nnWP, Sys.nnWP);
Sys.A = [PP, PI; IP, II];

end
