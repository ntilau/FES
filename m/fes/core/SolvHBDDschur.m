function [uF, Sys] = SolvHBDDschur(Sys)
% Solve the harmonic-balance DD Schur system with cached Schur complements.
%   (AFF + sum(SFprev{ir})) * uF = gF
%
% Only recomputes interior solves for regions where Sys.Compute(ir) is set.

SF = Sys.AFF * 0;

for ir = 1:length(Sys.RegDoF)
    if Sys.Compute(ir)
        Sys.SFprev{ir} = -Sys.AFI{ir} * (Sys.AII{ir} \ Sys.AIF{ir});
    end
    SF = SF + Sys.SFprev{ir};
end

SF = SF + Sys.AFF;
uF = SF \ Sys.gF;

end
