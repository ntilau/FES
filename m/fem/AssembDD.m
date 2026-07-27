function [Sys, Mesh] = AssembDD(Sys, Mesh, iReg)
% Assemble FE system matrices S and T for a single DD subdomain (iReg),
% including ABC/Dirichlet/DD boundary conditions on that subdomain.

tic;
Sys  = CalcDoFsNumber(Sys, Mesh);
Mesh = CalcDoFsPositions(Sys, Mesh);

FlagABC = false;
FlagDir = false;
FlagDD  = false;

if isfield(Mesh, 'BC')
    if isfield(Mesh.BC, 'ABC')
        idsABC    = find(Mesh.slab == Mesh.BC.ABC);
        Sys.idsABC = idsABC;
        FlagABC   = true;
    end
    if isfield(Mesh.BC, 'Dir')
        idsDir  = find(Mesh.slab == Mesh.BC.Dir);
        FlagDir = true;
    end
    if isfield(Mesh.BC, 'DD')
        idsDD     = find(Mesh.slab == Mesh.BC.DD);
        Sys.idsDD = idsDD;
        FlagDD    = true;
    end
end

%% Check materials
nMtrl = length(unique(Mesh.elab));
if ~isfield(Mesh, 'epsr')
    Mesh.epsr = ones(1, nMtrl);
end
if ~isfield(Mesh, 'mur')
    Mesh.mur = cell(1, nMtrl);
    for i = 1:nMtrl
        Mesh.mur{i} = eye(2);
    end
end

%% Quadrature and shape functions
num   = (Sys.pOrd + 1) * (Sys.pOrd + 2) / 2;
num2  = num^2;
[xq1, wq1]   = CalcSimplexQuad(Sys.pOrd + 1, 1);
[xyq2, wq2]  = CalcSimplexQuad(Sys.pOrd + 1, 2);
[Shape1, Shape1Deriv]       = CalcShapeFunctions(1, Sys.pOrd);
[Shape2, Shape2DerivX, Shape2DerivY] = CalcShapeFunctions(2, Sys.pOrd);

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

%% Element loop (only elements belonging to iReg)
II   = zeros(Mesh.NELE * num2, 1);
JJ   = II;
XXS  = II;
XXT  = II;
s    = 1;

for ie = 1:Mesh.NELE
    if Mesh.elab(ie) == iReg
        gIdx = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
        [detJ, invJt] = CalcJacobian(Mesh.node(Mesh.ele(ie, :), :));

        S  = zeros(num);
        T  = S;
        nu = Mesh.mur{Mesh.elab(ie)} \ eye(2);
        for iq = 1:length(wq2)
            S = S + detJ * ((invJt * dNQuad2{iq})' * nu ...
                * (invJt * dNQuad2{iq})) * wq2(iq);
            T = T + detJ * (NQuad2{iq}' * NQuad2{iq}) * wq2(iq) ...
                * Mesh.epsr(Mesh.elab(ie));
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
end

Sys.DirReg = unique(II(II > 0));
II(II == 0) = 1;
JJ(JJ == 0) = 1;
Sys.S = sparse(II, JJ, XXS, Sys.NDOFs, Sys.NDOFs);
Sys.T = sparse(II, JJ, XXT, Sys.NDOFs, Sys.NDOFs);
Sys.f = sparse(Sys.NDOFs, 1);

%% ABC boundary condition
if FlagABC
    Sys.fEinc = exp(-1i * Sys.k * (Sys.kEinc * Mesh.refNode.').');

    NSABC  = length(idsABC);
    IIABC  = ones(NSABC * (Sys.pOrd + 1)^2, 1);
    JJABC  = IIABC;
    XXABC  = IIABC - 1;
    s      = 1;
    for ie = 1:Mesh.NELE
        if Mesh.elab(ie) == iReg
            onABC = [sum(abs(Mesh.spig(ie, 1)) == idsABC) ...
                     sum(abs(Mesh.spig(ie, 2)) == idsABC) ...
                     sum(abs(Mesh.spig(ie, 3)) == idsABC)];
            if sum(onABC) > 0
                idOnABC = find(onABC == 1);
                for i = 1:length(idOnABC)
                    spigId  = Mesh.spig(ie, idOnABC(i));
                    nodeId  = Mesh.spig2(abs(spigId), :);
                    intNode = Mesh.ele(ie, ...
                        Mesh.ele(ie, :) ~= nodeId(1) ...
                        & Mesh.ele(ie, :) ~= nodeId(2));
                    gIdx = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnABC(i)).';
                    l    = diff(Mesh.node(nodeId, :));
                    n    = cross([l 0], cross([l 0], ...
                        [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                    n = n / norm(n);

                    TrBC = zeros(Sys.pOrd + 1);
                    for iq = 1:length(wq1)
                        TrBC = TrBC + norm(l) * (NQuad1{iq}' * NQuad1{iq}) * wq1(iq);
                    end

                    rho = [ones(length(wq1), 1) * Mesh.node(nodeId(1), :) ...
                           + xq1 * l, zeros(length(wq1), 1)];
                    dEinc = -1i * Sys.k * dot(Sys.kEinc, n) ...
                        * exp(-1i * Sys.k ...
                        * dot(Sys.kEinc.' * ones(1, Sys.pOrd + 1), rho.'));
                    frBC = zeros(Sys.pOrd + 1, 1);
                    for iq = 1:length(wq1)
                        frBC = frBC + norm(l) * (NQuad1{iq} .* dEinc).' * wq1(iq);
                    end

                    idx = s:s + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx      = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIABC(tmpIdx) = gIdx(j);
                            JJABC(tmpIdx) = gIdx(k);
                            XXABC(tmpIdx) = TrBC(j, k);
                        end
                    end
                    s = s + (Sys.pOrd + 1)^2;
                    Sys.f(gIdx) = Sys.f(gIdx) + frBC;
                end
            end
        end
    end
    Sys.ABC    = sparse(IIABC, JJABC, XXABC, Sys.NDOFs, Sys.NDOFs);
    Sys.DirABC = unique(IIABC(IIABC > 0));
end

%% DD interface condition
if FlagDD
    NSDD  = length(idsDD);
    IIDD  = ones(NSDD * (Sys.pOrd + 1)^2, 1);
    JJDD  = IIDD;
    XXDD  = IIDD - 1;
    s     = 1;
    for ie = 1:Mesh.NELE
        if Mesh.elab(ie) == iReg
            onDD = [sum(abs(Mesh.spig(ie, 1)) == idsDD) ...
                    sum(abs(Mesh.spig(ie, 2)) == idsDD) ...
                    sum(abs(Mesh.spig(ie, 3)) == idsDD)];
            if sum(onDD) > 0
                idOnDD = find(onDD == 1);
                for i = 1:length(idOnDD)
                    spigId  = Mesh.spig(ie, idOnDD(i));
                    nodeId  = Mesh.spig2(abs(spigId), :);
                    intNode = Mesh.ele(ie, ...
                        Mesh.ele(ie, :) ~= nodeId(1) ...
                        & Mesh.ele(ie, :) ~= nodeId(2));
                    gIdx = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDD(i)).';
                    l    = diff(Mesh.node(nodeId, :));
                    n    = cross([l 0], cross([l 0], ...
                        [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                    n = n / norm(n);

                    TrBC = zeros(Sys.pOrd + 1);
                    for iq = 1:length(wq1)
                        TrBC = TrBC + norm(l) * (NQuad1{iq}' * NQuad1{iq}) * wq1(iq);
                    end

                    rho = [ones(length(wq1), 1) * Mesh.node(nodeId(1), :) ...
                           + xq1 * l, zeros(length(wq1), 1)];
                    dEinc = -1i * Sys.k * dot(Sys.kEinc, n) ...
                        * exp(-1i * Sys.k ...
                        * dot(Sys.kEinc.' * ones(1, Sys.pOrd + 1), rho.'));
                    frBC = zeros(Sys.pOrd + 1, 1);
                    for iq = 1:length(wq1)
                        frBC = frBC + norm(l) * (NQuad1{iq} .* dEinc).' * wq1(iq);
                    end

                    idx = s:s + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx      = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIDD(tmpIdx) = gIdx(j);
                            JJDD(tmpIdx) = gIdx(k);
                            XXDD(tmpIdx) = TrBC(j, k);
                        end
                    end
                    s = s + (Sys.pOrd + 1)^2;
                    Sys.f(gIdx) = Sys.f(gIdx) + frBC;
                end
            end
        end
    end
    Sys.DD    = sparse(IIDD, JJDD, XXDD, Sys.NDOFs, Sys.NDOFs);
    Sys.DirDD = unique(IIDD(IIDD > 0));
end

%% Dirichlet condition
if FlagDir
    NSDir  = length(idsDir);
    IIDir  = zeros(NSDir * (Sys.pOrd + 1)^2, 1);
    s      = 1;
    for ie = 1:Mesh.NELE
        if Mesh.elab(ie) == iReg
            onDir = [sum(abs(Mesh.spig(ie, 1)) == idsDir) ...
                     sum(abs(Mesh.spig(ie, 2)) == idsDir) ...
                     sum(abs(Mesh.spig(ie, 3)) == idsDir)];
            if sum(onDir) > 0
                idOnDir = find(onDir == 1);
                for i = 1:length(idOnDir)
                    gIdx = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDir(i));
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
    end
    Sys.Dir = unique(IIDir(IIDir > 0));
end

fprintf('System matrices assembly: %2.4g s\n', toc);
end
