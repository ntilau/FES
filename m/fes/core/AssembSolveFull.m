function [SysF, MeshF, errorF, spF] = AssembSolveFull(SysF, MeshF)
% Direct solve driver for the full (non-DD) HB system with ferrite.
% Assembles the system, assembles the WP system, solves via SolvDir,
% and computes relative error and scattering parameters.

%% Assemble full system and solve
[SysF, MeshF] = AssembHBFerrite(SysF, MeshF);
SysF = AssembWPHB(SysF);
XF   = SolvDir(SysF);

%% Reconstruct full solution
SysF.u = zeros(SysF.NDOFs * SysF.nHarms, 1);
SysF.u(SysF.nnWP) = XF(length(SysF.WP) * SysF.WPnModes ...
    * SysF.nHarms + 1:end, 1);

for jh = 1:length(SysF.Harms)
    for ip = 1:length(SysF.WP)
        SysF.u((jh - 1) * SysF.NDOFs + SysF.WP{ip}) = SysF.WPgvec{ip, jh} ...
            * (XF((1:SysF.WPnModes) ...
                  + ((jh - 1) + (ip - 1) * SysF.nHarms) * SysF.WPnModes, 1) ...
               .* SysF.Power((1:SysF.WPnModes) ...
                  + ((jh - 1) + (ip - 1) * SysF.nHarms) * SysF.WPnModes).');
    end
end

%% Error and scattering parameters
errorF = norm(SysF.u - SysF.u0) / norm(SysF.u);
SysF.u0 = SysF.u;

spF = XF(1:length(SysF.WP) * SysF.WPnModes * SysF.nHarms, 1);
spF(1, 1) = (spF(1, 1) - 1) * sqrt(SysF.Pfund);
spF(SysF.WPnum, 1) = (spF(SysF.WPnum, 1) - 1) * sqrt(SysF.Pitrf);

end
