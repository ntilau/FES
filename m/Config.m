%% Config - Initialize paths and logging
function Config()
    addpath(genpath('.'));
    Sys.log = sprintf('#Begin: %d/%d/%d, %d:%d:%.2g\n', clock);
end
