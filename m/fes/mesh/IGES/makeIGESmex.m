function makeIGESmex()
% Makefile
% run
% >> makeIGESmex
% in MATLAB to compile the source code in the IGES-toolbox


try
    mex -v nrbevalIGES.c
end

try
    mex -v nrbSrfRegularEvalIGES.c
end

try
    mex -v closestNrbLinePointIGES.c
end

try
    mex -v nrbCrvPlaneIntrsctIGES.c
end

try
    mex -v createDVGtree.c
end

try
    mex -v icpDVG.c
end
