function [Sys, Mesh, errorD, spD] = AssembSolveDD(Sys, Mesh)
% Iterative solve driver for the HB-DD Schur system with ferrite update.
% Assembles the nonlinear ferrite region, assembles the WP system,
% solves via SolvHBDDschur, and computes relative error vs u0.

%% Initialization: assemble linear part (all regions except NLlab)
if ~isfield(Sys, 'first')
    Sys.RegToCompute = 1:Mesh.NLlab - 1;
    [Sys, Mesh] = AssembHBDDFerrite(Sys, Mesh);
    Sys.Slin    = Sys.S;
    Sys.Tlin    = Sys.T;
    Sys.first   = true;
end

%% Assemble nonlinear region and combine with linear part
Sys.RegToCompute = Mesh.NLlab;
[Sys, Mesh] = AssembHBDDFerrite(Sys, Mesh);
Sys.S       = Sys.S + Sys.Slin;
Sys.T       = Sys.T + Sys.Tlin;

%% Assemble WP system and solve
Sys = AssembWPHBDDschur(Sys);
[XD, Sys] = SolvHBDDschur(Sys);

%% Reconstruct full solution
Sys.Compute = Sys.Compute * 0;
Sys.Compute(Mesh.NLlab) = 1;

Sys.u = zeros(Sys.NDOFs * Sys.nHarms, 1);

bndIdx = length(Sys.WP) * Sys.WPnModes * Sys.nHarms + 1;
Sys.u(Sys.nnWP(Sys.BndDoFRed(bndIdx:end) - length(Sys.WP) ...
    * Sys.WPnModes * Sys.nHarms)) = XD(bndIdx:end, 1);

for jh = 1:length(Sys.Harms)
    for ip = 1:length(Sys.WP)
        Sys.u((jh - 1) * Sys.NDOFs + Sys.WP{ip}) = Sys.WPgvec{ip, jh} ...
            * XD((1:Sys.WPnModes) ...
                 + ((jh - 1) + (ip - 1) * Sys.nHarms) * Sys.WPnModes, 1);
    end
end

%% Reconstruct interior region
X5 = Sys.AII{Mesh.NLlab} \ (-Sys.AIF{Mesh.NLlab} * XD(:, 1));
Sys.u(Sys.nnWP(Sys.RegDoFRed{Mesh.NLlab} ...
    - length(Sys.WP) * Sys.WPnModes * Sys.nHarms)) = X5(:, 1);

%% Error and scattering parameters
errorD = norm(Sys.u - Sys.u0) / norm(Sys.u);
Sys.u0 = Sys.u;

spD = XD(1:length(Sys.WP) * Sys.WPnModes * Sys.nHarms, 1);
spD(1, 1) = (spD(1, 1) - 1) * sqrt(Sys.Pfund);
spD(Sys.WPnum, 1) = (spD(Sys.WPnum, 1) - 1) * sqrt(Sys.Pitrf);

end
