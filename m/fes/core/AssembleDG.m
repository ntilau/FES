function [Sys, Mesh] = AssembleDG(Sys, Mesh)
% Assemble FE system matrices using the Discontinuous Galerkin (DG)
% formulation; includes local element DoFs, Dirichlet, and Neumann BCs.
%
% NOTE: This file uses local (non-prefixed) function names for CalcGlobIndex,
% CalcShapeFunctions, etc. as it is a standalone variant.

tic;
Sys.NDOF = DoFsNumber(Sys.pOrd, Mesh.NNODE, Mesh.NELE, Mesh.NSPIG);
fprintf('--> h = %d, p = %d, NDOF = %d\n', Sys.hOrd, Sys.pOrd, Sys.NDOF);
Mesh = DoFsPositions(Sys, Mesh);

FlagDir = false;
FlagNeu = false;

if isfield(Mesh, 'BC')
    if isfield(Mesh.BC, 'Dir')
        idsDir = cell(length(Mesh.BC.Dir), 1);
        for ibc = 1:length(Mesh.BC.Dir)
            idsDir{ibc} = find(Mesh.slab == Mesh.BC.Dir(ibc));
        end
        FlagDir = true;
    end
    if isfield(Mesh.BC, 'Neu')
        idsNeu  = find(Mesh.slab == Mesh.BC.Neu);
        FlagNeu = true;
    end
end

num  = (Sys.pOrd + 1) * (Sys.pOrd + 2) / 2;
num2 = num^2;

[xq1, wq1]  = SimplexQuad(Sys.pOrd + 1, 1);
[xyq2, wq2] = SimplexQuad(Sys.pOrd + 1, 2);
[Shape1, Shape1Deriv]       = ShapeFunctions(1, Sys.pOrd);
[Shape2, Shape2DerivX, Shape2DerivY] = ShapeFunctions(2, Sys.pOrd);

NQuad1  = cell(1, length(wq1));
dNQuad1 = cell(1, length(wq1));
for iq = 1:length(wq1)
    NQuad1{iq}  = Shape1(xq1(iq));
    dNQuad1{iq} = Shape1Deriv(xq1(iq));
end

NQuad2  = cell(1, length(wq2));
dNQuad2 = cell(1, length(wq2));
for iq = 1:length(wq2)
    NQuad2{iq} = Shape2(xyq2(iq, 1), xyq2(iq, 2));
    dNQuad2{iq} = [Shape2DerivX(xyq2(iq, 1), xyq2(iq, 2)); ...
                   Shape2DerivY(xyq2(iq, 1), xyq2(iq, 2))];
end

%% Element loop
II   = zeros(Mesh.NELE * num2, 1);
JJ   = II;
XXS  = II;
XXT  = II;
s    = 1;

for ie = 1:Mesh.NELE
    gIdx = GlobIndex(2, Sys.pOrd, Mesh, ie);
    [detJ, invJt] = Jacobian(Mesh.node(Mesh.ele(ie, :), :));

    S = zeros(num);
    T = S;
    for iq = 1:length(wq2)
        S = S + detJ * ((invJt * dNQuad2{iq})' * (invJt * dNQuad2{iq})) * wq2(iq);
        T = T + detJ * (NQuad2{iq}' * NQuad2{iq}) * wq2(iq);
    end

    idx = s:s + num2 - 1;
    for j = 1:num
        for k = 1:num
            tmpIdx     = idx(num * (j - 1) + k);
            II(tmpIdx)  = gIdx(j);
            JJ(tmpIdx)  = gIdx(k);
            XXS(tmpIdx) = S(j, k);
            XXT(tmpIdx) = T(j, k);
        end
    end
    s = s + num2;
end

Sys.S = sparse(II, JJ, XXS, Sys.NDOF, Sys.NDOF);
Sys.T = sparse(II, JJ, XXT, Sys.NDOF, Sys.NDOF);
Sys.f = sparse(Sys.NDOF, 1);

%% Dirichlet boundary condition
if FlagDir
    for ibc = 1:length(idsDir)
        NSDir  = length(idsDir{ibc});
        IIDir  = zeros(NSDir * (Sys.pOrd + 1)^2, 1);
        s      = 1;
        for ie = 1:Mesh.NELE
            onDir = [sum(abs(Mesh.spig(ie, 1)) == idsDir{ibc}) ...
                     sum(abs(Mesh.spig(ie, 2)) == idsDir{ibc}) ...
                     sum(abs(Mesh.spig(ie, 3)) == idsDir{ibc})];
            if sum(onDir) > 0
                idOnDir = find(onDir == 1);
                for i = 1:length(idOnDir)
                    gIdx = GlobIndex(1, Sys.pOrd, Mesh, ie, idOnDir(i));
                    idx = s:s + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx      = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIDir(tmpIdx) = gIdx(j);
                        end
                    end
                    s = s + (Sys.pOrd + 1)^2;
                end
            end
        end
        Sys.Dir{ibc} = unique(IIDir(IIDir > 0));
    end
end

%% Neumann boundary condition
if FlagNeu
    NSNeu  = length(idsNeu);
    IINeu  = zeros(NSNeu * (Sys.pOrd + 1)^2, 1);
    s      = 1;
    for ie = 1:Mesh.NELE
        onNeu = [sum(abs(Mesh.spig(ie, 1)) == idsNeu) ...
                 sum(abs(Mesh.spig(ie, 2)) == idsNeu) ...
                 sum(abs(Mesh.spig(ie, 3)) == idsNeu)];
        if sum(onNeu) > 0
            idOnNeu = find(onNeu == 1);
            for i = 1:length(idOnNeu)
                spigId  = abs(Mesh.spig(ie, idOnNeu(i)));
                nodeId  = Mesh.spig2(spigId, :);
                intNode = Mesh.ele(ie, ...
                    Mesh.ele(ie, :) ~= nodeId(1) ...
                    & Mesh.ele(ie, :) ~= nodeId(2));
                gIdx = GlobIndex(1, Sys.pOrd, Mesh, ie, idOnNeu(i));
                l    = diff(Mesh.node(nodeId, :));
                n    = cross([l 0], cross([l 0], ...
                    [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                n = n / norm(n);
                frBC = zeros(Sys.pOrd + 1, 1);
                for iq = 1:length(wq1)
                    frBC = frBC + norm(l) * (NQuad1{iq}).' * wq1(iq);
                end
                idx = s:s + (Sys.pOrd + 1)^2 - 1;
                for j = 1:(Sys.pOrd + 1)
                    for k = 1:(Sys.pOrd + 1)
                        tmpIdx       = idx((Sys.pOrd + 1) * (j - 1) + k);
                        IINeu(tmpIdx) = gIdx(j);
                    end
                end
                s = s + (Sys.pOrd + 1)^2;
                Sys.f(gIdx) = Sys.f(gIdx) + frBC;
            end
        end
    end
    Sys.Neu = unique(IINeu(IINeu > 0));
end

fprintf('System matrices assembly: %2.4g s\n', toc);

end

% -------------------------------------------------------------------------
% Local helper functions (standalone, not prefixed with Calc*)
% -------------------------------------------------------------------------

function NDOF = DoFsNumber(pOrd, NNODE, NELE, NSPIG)
NDOF = NNODE + NSPIG * (pOrd - 1) ...
     + NELE * (0 < (pOrd - 2)) * (pOrd - 2) * (pOrd - 1) * .5;
end

function Mesh = DoFsPositions(Sys, Mesh)
Mesh.refNode = zeros(Sys.NDOF, 3);
Mesh.refEle  = zeros(Mesh.NELE * Sys.pOrd^2, 3);
for ie = 1:Mesh.NELE
    gIdx = GlobIndex(2, Sys.pOrd, Mesh, ie);
    nIdx = Mesh.ele(ie, :);
    sIdx = abs(Mesh.spig(ie, :));
    Mesh.refNode(gIdx(1:3), :) = [Mesh.node(nIdx, :) zeros(3, 1)];
    switch Sys.pOrd
        case 1
            Mesh.refEle(ie, :) = nIdx;
        case 2
            nIdx = Mesh.spig2(sIdx, :);
            Mesh.refNode(gIdx(4:6), :) = [1/2 * Mesh.node(nIdx(:, 1), :) ...
                + 1/2 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            Mesh.refEle(4 * (ie - 1) + 1, :) = gIdx([1 5 6]);
            Mesh.refEle(4 * (ie - 1) + 2, :) = gIdx([4 5 6]);
            Mesh.refEle(4 * (ie - 1) + 3, :) = gIdx([2 4 6]);
            Mesh.refEle(4 * (ie - 1) + 4, :) = gIdx([3 4 5]);
        case 3
            nIdx = Mesh.spig2(sIdx, :);
            Mesh.refNode(gIdx(4:6), :)  = [2/3 * Mesh.node(nIdx(:, 1), :) ...
                + 1/3 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            Mesh.refNode(gIdx(7:9), :)  = [1/3 * Mesh.node(nIdx(:, 1), :) ...
                + 2/3 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            nIdx = Mesh.ele(ie, :);
            Mesh.refNode(gIdx(10), :) = [1/3 * Mesh.node(nIdx(1), :) ...
                + 1/3 * Mesh.node(nIdx(2), :) ...
                + 1/3 * Mesh.node(nIdx(3), :) 0];
            Mesh.refEle(9 * (ie - 1) + 1, :) = gIdx([1 6 5]);
            Mesh.refEle(9 * (ie - 1) + 2, :) = gIdx([6 5 10]);
            Mesh.refEle(9 * (ie - 1) + 3, :) = gIdx([6 9 10]);
            Mesh.refEle(9 * (ie - 1) + 4, :) = gIdx([4 9 10]);
            Mesh.refEle(9 * (ie - 1) + 5, :) = gIdx([2 4 9]);
            Mesh.refEle(9 * (ie - 1) + 6, :) = gIdx([4 7 10]);
            Mesh.refEle(9 * (ie - 1) + 7, :) = gIdx([8 7 10]);
            Mesh.refEle(9 * (ie - 1) + 8, :) = gIdx([5 8 10]);
            Mesh.refEle(9 * (ie - 1) + 9, :) = gIdx([3 8 7]);
        case 4
            nIdx = Mesh.spig2(sIdx, :);
            Mesh.refNode(gIdx(4:6), :)    = [3/4 * Mesh.node(nIdx(:, 1), :) ...
                + 1/4 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            Mesh.refNode(gIdx(7:9), :)    = [2/4 * Mesh.node(nIdx(:, 1), :) ...
                + 2/4 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            Mesh.refNode(gIdx(11:13), :)  = [1/4 * Mesh.node(nIdx(:, 1), :) ...
                + 3/4 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            nIdx = Mesh.ele(ie, :);
            Mesh.refNode(gIdx(10), :) = [1/2 * Mesh.node(nIdx(1), :) ...
                + 1/4 * Mesh.node(nIdx(2), :) ...
                + 1/4 * Mesh.node(nIdx(3), :) 0];
            Mesh.refNode(gIdx(14), :) = [1/4 * Mesh.node(nIdx(1), :) ...
                + 1/2 * Mesh.node(nIdx(2), :) ...
                + 1/4 * Mesh.node(nIdx(3), :) 0];
            Mesh.refNode(gIdx(15), :) = [1/4 * Mesh.node(nIdx(1), :) ...
                + 1/4 * Mesh.node(nIdx(2), :) ...
                + 1/2 * Mesh.node(nIdx(3), :) 0];
            Mesh.refEle(16 * (ie - 1) + 1, :)  = gIdx([1 5 6]);
            Mesh.refEle(16 * (ie - 1) + 2, :)  = gIdx([5 6 10]);
            Mesh.refEle(16 * (ie - 1) + 3, :)  = gIdx([6 9 10]);
            Mesh.refEle(16 * (ie - 1) + 4, :)  = gIdx([9 10 14]);
            Mesh.refEle(16 * (ie - 1) + 5, :)  = gIdx([9 13 14]);
            Mesh.refEle(16 * (ie - 1) + 6, :)  = gIdx([4 13 14]);
            Mesh.refEle(16 * (ie - 1) + 7, :)  = gIdx([2 4 13]);
            Mesh.refEle(16 * (ie - 1) + 8, :)  = gIdx([4 7 14]);
            Mesh.refEle(16 * (ie - 1) + 9, :)  = gIdx([7 14 15]);
            Mesh.refEle(16 * (ie - 1) + 10, :) = gIdx([10 14 15]);
            Mesh.refEle(16 * (ie - 1) + 11, :) = gIdx([8 10 15]);
            Mesh.refEle(16 * (ie - 1) + 12, :) = gIdx([8 10 5]);
            Mesh.refEle(16 * (ie - 1) + 13, :) = gIdx([8 12 15]);
            Mesh.refEle(16 * (ie - 1) + 14, :) = gIdx([11 12 15]);
            Mesh.refEle(16 * (ie - 1) + 15, :) = gIdx([7 11 15]);
            Mesh.refEle(16 * (ie - 1) + 16, :) = gIdx([3 11 12]);
    end
end
Mesh.refEle = sort(Mesh.refEle.').';
end

function gIdx = GlobIndex(dim, pOrd, Mesh, ie, is)
NNODE = Mesh.NNODE;
NELE  = Mesh.NELE;
NSPIG = Mesh.NSPIG;
switch dim
    case 1
        sIdx = abs(Mesh.spig(ie, is));
        nIdx = Mesh.spig2(sIdx, :);
        switch pOrd
            case 1
                gIdx = nIdx;
            case 2
                gIdx = [nIdx NNODE + abs(sIdx)];
            case 3
                gIdx = [nIdx NNODE + abs(sIdx) NNODE + NSPIG + abs(sIdx)];
            case 4
                gIdx = [nIdx NNODE + abs(sIdx) ...
                        NNODE + NSPIG + abs(sIdx) ...
                        NNODE + 2 * NSPIG + NELE + abs(sIdx)];
        end
    case 2
        ele  = Mesh.ele(ie, :);
        spig = abs(Mesh.spig(ie, :));
        switch pOrd
            case 1
                gIdx = ele;
                gIv  = abs(Mesh.spig(ie, :));
            case 2
                gIdx = [ele NNODE + spig];
            case 3
                gIdx = [ele NNODE + spig ...
                        NNODE + NSPIG + spig NNODE + 2 * NSPIG + ie];
            case 4
                gIdx = [ele NNODE + spig ...
                        NNODE + NSPIG + spig NNODE + 2 * NSPIG + ie ...
                        NNODE + 2 * NSPIG + NELE + spig ...
                        NNODE + 3 * NSPIG + NELE + ie ...
                        NNODE + 3 * NSPIG + 2 * NELE + ie];
        end
end
end

function [detJ, invJt] = Jacobian(xy)
J    = [xy(2, 1) - xy(1, 1) xy(3, 1) - xy(1, 1); ...
        xy(2, 2) - xy(1, 2) xy(3, 2) - xy(1, 2)];
detJ = abs(det(J));
invJt = inv(J)';
end

% -------------------------------------------------------------------------
% Shape functions (inline-based, standalone form)
% -------------------------------------------------------------------------
function [Shape, ShapeDerivX, ShapeDerivY] = ShapeFunctions(dim, p)
Shape       = inline('');
ShapeDerivX = inline('');
ShapeDerivY = inline('');
switch dim
    case 1
        switch p
            case 1
                Shape       = inline('[ 1 - x, x]', 'x');
                ShapeDerivX = inline('[-1, 1]', 'x');
            case 2
                Shape       = inline('[ (2*x - 1)*(x - 1), x*(2*x - 1), -4*x*(x - 1)]', 'x');
                ShapeDerivX = inline('[ 4*x - 3, 4*x - 1, 4 - 8*x]', 'x');
            case 3
                Shape       = inline('[ -((3*x - 1)*(3*x - 2)*(x - 1))/2, (x*(3*x - 1)*(3*x - 2))/2, (9*x*(3*x - 2)*(x - 1))/2, -(9*x*(3*x - 1)*(x - 1))/2]', 'x');
                ShapeDerivX = inline('[ 18*x - (27*x^2)/2 - 11/2, (27*x^2)/2 - 9*x + 1, (81*x^2)/2 - 45*x + 9, 36*x - (81*x^2)/2 - 9/2]', 'x');
            case 4
                Shape       = inline('[ ((2*x - 1)*(4*x - 1)*(4*x - 3)*(x - 1))/3, (x*(2*x - 1)*(4*x - 1)*(4*x - 3))/3, -(16*x*(2*x - 1)*(4*x - 3)*(x - 1))/3, 4*x*(4*x - 1)*(4*x - 3)*(x - 1), -(16*x*(2*x - 1)*(4*x - 1)*(x - 1))/3]', 'x');
                ShapeDerivX = inline('[ ((8*x - 5)*(16*x^2 - 20*x + 5))/3, ((8*x - 3)*(16*x^2 - 12*x + 1))/3, - (512*x^3)/3 + 288*x^2 - (416*x)/3 + 16, 4*(2*x - 1)*(32*x^2 - 32*x + 3), - (512*x^3)/3 + 224*x^2 - (224*x)/3 + 16/3]', 'x');
        end
    case 2
        switch p
            case 1
                Shape        = inline('[ 1 - y - x, x, y]', 'x', 'y');
                ShapeDerivX  = inline('[-1, 1, 0]', 'x', 'y');
                ShapeDerivY  = inline('[-1, 0, 1]', 'x', 'y');
            case 2
                Shape        = inline('[ (2*x + 2*y - 1)*(x + y - 1), x*(2*x - 1), y*(2*y - 1), 4*x*y, -4*y*(x + y - 1), -4*x*(x + y - 1)]', 'x', 'y');
                ShapeDerivX  = inline('[ 4*x + 4*y - 3, 4*x - 1, 0, 4*y, -4*y, 4 - 4*y - 8*x]', 'x', 'y');
                ShapeDerivY  = inline('[ 4*x + 4*y - 3, 0, 4*y - 1, 4*x, 4 - 8*y - 4*x, -4*x]', 'x', 'y');
            case 3
                Shape        = inline('[ -((3*x + 3*y - 1)*(3*x + 3*y - 2)*(x + y - 1))/2, (x*(3*x - 1)*(3*x - 2))/2, (y*(3*y - 1)*(3*y - 2))/2, (9*x*y*(3*x - 1))/2, (9*y*(3*x + 3*y - 2)*(x + y - 1))/2, (9*x*(3*x + 3*y - 2)*(x + y - 1))/2, (9*x*y*(3*y - 1))/2, -(9*y*(3*y - 1)*(x + y - 1))/2, -(9*x*(3*x - 1)*(x + y - 1))/2, -27*x*y*(x + y - 1)]', 'x', 'y');
                ShapeDerivX  = inline('[ 18*x + 18*y - 27*x*y - (27*x^2)/2 - (27*y^2)/2 - 11/2, (27*x^2)/2 - 9*x + 1, 0, (9*y*(6*x - 1))/2, (9*y*(6*x + 6*y - 5))/2, (81*x^2)/2 + 54*x*y - 45*x + (27*y^2)/2 - (45*y)/2 + 9, (9*y*(3*y - 1))/2, -(9*y*(3*y - 1))/2, 36*x + (9*y)/2 - 27*x*y - (81*x^2)/2 - 9/2, -27*y*(2*x + y - 1)]', 'x', 'y');
                ShapeDerivY  = inline('[ 18*x + 18*y - 27*x*y - (27*x^2)/2 - (27*y^2)/2 - 11/2, 0, (27*y^2)/2 - 9*y + 1, (9*x*(3*x - 1))/2, (27*x^2)/2 + 54*x*y - (45*x)/2 + (81*y^2)/2 - 45*y + 9, (9*x*(6*x + 6*y - 5))/2, (9*x*(6*y - 1))/2, (9*x)/2 + 36*y - 27*x*y - (81*y^2)/2 - 9/2, -(9*x*(3*x - 1))/2, -27*x*(x + 2*y - 1)]', 'x', 'y');
            case 4
                Shape        = inline('[ ((2*x + 2*y - 1)*(4*x + 4*y - 1)*(4*x + 4*y - 3)*(x + y - 1))/3, (x*(2*x - 1)*(4*x - 1)*(4*x - 3))/3, (y*(2*y - 1)*(4*y - 1)*(4*y - 3))/3, (16*x*y*(2*x - 1)*(4*x - 1))/3, -(16*y*(2*x + 2*y - 1)*(4*x + 4*y - 3)*(x + y - 1))/3, -(16*x*(2*x + 2*y - 1)*(4*x + 4*y - 3)*(x + y - 1))/3, 4*x*y*(4*x - 1)*(4*y - 1), 4*y*(4*y - 1)*(4*x + 4*y - 3)*(x + y - 1), 4*x*(4*x - 1)*(4*x + 4*y - 3)*(x + y - 1), 32*x*y*(4*x + 4*y - 3)*(x + y - 1), (16*x*y*(2*y - 1)*(4*y - 1))/3, -(16*y*(2*y - 1)*(4*y - 1)*(x + y - 1))/3, -(16*x*(2*x - 1)*(4*x - 1)*(x + y - 1))/3, -32*x*y*(4*x - 1)*(x + y - 1), -32*x*y*(4*y - 1)*(x + y - 1)]', 'x', 'y');
                ShapeDerivX  = inline('[ ((8*x + 8*y - 5)*(16*x^2 + 32*x*y - 20*x + 16*y^2 - 20*y + 5))/3, ((8*x - 3)*(16*x^2 - 12*x + 1))/3, 0, (16*y*(24*x^2 - 12*x + 1))/3, -(16*y*(24*x^2 + 48*x*y - 36*x + 24*y^2 - 36*y + 13))/3, - (512*x^3)/3 - 384*x^2*y + 288*x^2 - 256*x*y^2 + 384*x*y - (416*x)/3 - (128*y^3)/3 + 96*y^2 - (208*y)/3 + 16, 4*y*(8*x - 1)*(4*y - 1), 4*y*(4*y - 1)*(8*x + 8*y - 7), 4*(2*x + y - 1)*(32*x*y - 4*y - 32*x + 32*x^2 + 3), 32*y*(12*x^2 + 16*x*y - 14*x + 4*y^2 - 7*y + 3), (16*y*(2*y - 1)*(4*y - 1))/3, -(16*y*(2*y - 1)*(4*y - 1))/3, 64*x*y - (16*y)/3 - (224*x)/3 - 128*x^2*y + 224*x^2 - (512*x^3)/3 + 16/3, -32*y*(8*x*y - y - 10*x + 12*x^2 + 1), -32*y*(4*y - 1)*(2*x + y - 1)]', 'x', 'y');
                ShapeDerivY  = inline('[ ((8*x + 8*y - 5)*(16*x^2 + 32*x*y - 20*x + 16*y^2 - 20*y + 5))/3, 0, ((8*y - 3)*(16*y^2 - 12*y + 1))/3, (16*x*(2*x - 1)*(4*x - 1))/3, - (128*x^3)/3 - 256*x^2*y + 96*x^2 - 384*x*y^2 + 384*x*y - (208*x)/3 - (512*y^3)/3 + 288*y^2 - (416*y)/3 + 16, -(16*x*(24*x^2 + 48*x*y - 36*x + 24*y^2 - 36*y + 13))/3, 4*x*(4*x - 1)*(8*y - 1), 4*(x + 2*y - 1)*(32*x*y - 32*y - 4*x + 32*y^2 + 3), 4*x*(4*x - 1)*(8*x + 8*y - 7), 32*x*(4*x^2 + 16*x*y - 7*x + 12*y^2 - 14*y + 3), (16*x*(24*y^2 - 12*y + 1))/3, 64*x*y - (224*y)/3 - (16*x)/3 - 128*x*y^2 + 224*y^2 - (512*y^3)/3 + 16/3, -(16*x*(2*x - 1)*(4*x - 1))/3, -32*x*(4*x - 1)*(x + 2*y - 1), -32*x*(8*x*y - 10*y - x + 12*y^2 + 1)]', 'x', 'y');
        end
end
end

function [X, W] = SimplexQuad(varargin)
% Simplex quadrature rules (Gauss-Legendre) for n-dimensional simplices.
% Greg von Winckel -- http://math.unm.edu/~gregvw

nargin = length(varargin);
if nargin ~= 2
    error('simplexquad takes two input arguments');
end
N = varargin{1};
if length(N) ~= 1
    error('First argument must be a scalar');
end
if N ~= abs(round(N - 1)) + 1
    error('Number of Gauss points must be a natural number');
end

if length(varargin{2}) == 1  % Dimension specified
    n = varargin{2};
    if n ~= abs(round(n - 1)) + 1
        error('Dimension must be a natural number');
    end
    m = n + 1; vert = eye(m, n);
else                          % Vertices specified
    vert = varargin{2};
    [m, n] = size(vert);
    if m ~= n + 1
        error('Matrix of vertices must have n+1 rows and n columns');
    end
end

Nn = N^n;
if n == 1
    [q, w] = rquad(N, 0);
    len    = diff(vert);
    X      = vert(1) + len * q;
    W      = abs(len) * w;
else
    for k = 1:n
        [q{k}, w{k}] = rquad(N, n - k);
    end
    [Q{1:n}] = ndgrid(q{:});
    q = reshape(cat(n, Q{:}), Nn, n);
    [W{1:n}] = ndgrid(w{:});
    w = reshape(cat(n, W{:}), Nn, n);
    map = eye(m);
    map(2:m, 1) = -1;
    c = map * vert;
    W = abs(det(c(2:m, :))) * prod(w, 2);
    qp = cumprod(q, 2);
    e  = ones(Nn, 1);
    X  = [e, [(1 - q(:, 1:n - 1)), e] .* [e, qp(:, 1:n - 2), qp(:, n)]] * c;
end
end

function [x, w] = rquad(N, k)
k1 = k + 1; k2 = k + 2;
n  = 1:N; nnk = 2 * n + k;
A  = [k / k2, repmat(k^2, 1, N) ./ (nnk .* (nnk + 2))];
n  = 2:N; nnk = nnk(n);
B1 = 4 * k1 / (k2 * k2 * (k + 3));
nk = n + k; nnk2 = nnk .* nnk;
B  = 4 * (n .* nk).^2 ./ (nnk2 .* nnk2 - nnk2);
ab = [A', [(2^k1) / k1; B1; B']];
s  = sqrt(ab(2:N, 2));
[V, X] = eig(diag(ab(1:N, 1), 0) + diag(s, -1) + diag(s, 1));
[X, I] = sort(diag(X));
x = (X + 1) / 2;
w = (1 / 2)^(k1) * ab(1, 2) * V(1, I)'.^2;
end
