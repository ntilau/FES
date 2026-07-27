function uF = SolvDDschur(Sys)
% Solve the Schur-complement system for domain decomposition:
%   (AFF - sum(AFI * AII^{-1} * AIF)) * uF = gF
%
% Returns boundary DoF solution uF.

SF = Sys.AFF * 0;

% Compute the Schur complement by looping over subdomains
for ir = 1:length(Sys.RegDoF)
    fprintf('%g\n', length(Sys.AII{ir}));
    tic
    SF = SF - Sys.AFI{ir} * (Sys.AII{ir} \ Sys.AIF{ir});
    toc
end

% Assemble and solve
tic
SF = SF + Sys.AFF;
uF = SF \ Sys.gF;
toc

end
