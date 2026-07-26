%% ThermalFE - ODE right-hand side for thermal transient problem
function u = ThermalFE(t, u1)

global Sys;

A = Sys.T;
b = (Sys.q0 * Sys.f - Sys.K * Sys.S * u1) / Sys.RhoCp;

%% Apply Dirichlet boundary conditions
if isfield(Sys, 'Dir')
    for ibc = 1:length(Sys.Dir)
        b = b - A(:, Sys.Dir{ibc}) ...
            * ones(length(Sys.Dir{ibc}), 1) * Sys.Tedge{ibc};
    end
    for ibc = 1:length(Sys.Dir)
        b(Sys.Dir{ibc}) = Sys.Tedge{ibc};
        A(Sys.Dir{ibc}, :) = 0;
        A(:, Sys.Dir{ibc}) = 0;
        A(Sys.Dir{ibc}, Sys.Dir{ibc}) = eye(length(Sys.Dir{ibc}));
    end
end

u = A \ b;

end
