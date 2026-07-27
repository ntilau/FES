%% Config - Initialize paths and logging
% Directory layout: fes/ (assembly), mesh/ (mesh I/O + IGES + iormesh-src),
%                   post/ (VTK export), projects/ (drivers), tests/ (test scripts).
function Config()
    addpath(genpath('.'));
    Sys.log = sprintf('#Begin: %d/%d/%d, %d:%d:%.2g\n', clock);
end
