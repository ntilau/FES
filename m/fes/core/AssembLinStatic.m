function [Sys, Mesh] = AssembLinStatic(Sys, Mesh)
% Assemble linear FE matrices for the static (electrostatic) case,
% using epsr in the stiffness term (rather than mur).

FlagABC = false;
FlagDir = false;
FlagNeu = false;
FlagDD  = false;
FlagWP  = false;

if ~isfield(Sys, 'bypass')
    %% Constants
    Sys = GetConstants(Sys);

    %% Raw DoF numbering and positions
    Sys  = CalcDoFsNumber(Sys, Mesh);
    Mesh = CalcDoFsPositions(Sys, Mesh);

    %% Check boundary conditions
    if isfield(Mesh, 'BC')
        if isfield(Mesh.BC, 'ABC')
            idsABC    = find(Mesh.slab == Mesh.BC.ABC);
            Sys.idsABC = idsABC;
            FlagABC   = true;
        end
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
        if isfield(Mesh.BC, 'DD')
            idsDD     = find(Mesh.slab == Mesh.BC.DD);
            Sys.idsDD = idsDD;
            FlagDD    = true;
        end
        if isfield(Mesh.BC, 'WP')
            idsWP = cell(length(Mesh.BC.WP), 1);
            for ibc = 1:length(Mesh.BC.WP)
                idsWP{ibc} = find(Mesh.slab == Mesh.BC.WP(ibc));
            end
            FlagWP = true;
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
end

%% Master matrix dimensions
[nums, numv] = CalcOrderMatSize(Sys.pOrd);

%% Quadrature points
[xq1, wq1]   = CalcSimplexQuad(Sys.pOrd + 1, 1);
[xyq2, wq2]  = CalcSimplexQuad(Sys.pOrd + 1, 2);

%% Shape functions
vShap = false;
if isfield(Sys, 'Hcurl')
    vShap = true;
end
[Shape1, Shape1Deriv]       = CalcShapeFunctions(1, Sys.pOrd);
[Shape2, Shape2DerivX, Shape2DerivY] = CalcShapeFunctions(2, Sys.pOrd);

% Scalar -- 1-D quadrature
Ns1  = cell(1, length(wq1));
dNs1 = cell(1, length(wq1));
for iq = 1:length(wq1)
    Ns1{iq}  = Shape1(xq1(iq));
    dNs1{iq} = Shape1Deriv(xq1(iq));
end

% Scalar -- 2-D quadrature
Ns2  = cell(1, length(wq2));
dNs2 = cell(1, length(wq2));
for iq = 1:length(wq2)
    Ns2{iq} = Shape2(xyq2(iq, 1), xyq2(iq, 2));
    dNs2{iq} = [Shape2DerivX(xyq2(iq, 1), xyq2(iq, 2)); ...
                Shape2DerivY(xyq2(iq, 1), xyq2(iq, 2))];
end

% Vector (H(curl)) -- 2-D quadrature at p=1
if vShap
    [Shape2_1, Shape2DerivX_1, Shape2DerivY_1] = CalcShapeFunctions(2, 1);
    Ns2_1  = cell(1, length(wq2));
    dNs2_1 = cell(1, length(wq2));
    for iq = 1:length(wq2)
        Ns2_1{iq} = Shape2_1(xyq2(iq, 1), xyq2(iq, 2));
        dNs2_1{iq} = [Shape2DerivX_1(xyq2(iq, 1), xyq2(iq, 2)); ...
                      Shape2DerivY_1(xyq2(iq, 1), xyq2(iq, 2))];
    end
end

%% Element loop
IIs  = zeros(Mesh.NELE * nums^2, 1);
JJs  = IIs;
XXSs = IIs;
XXTs = IIs;
is   = 1;

if vShap
    IIv   = zeros(Mesh.NELE * numv^2, 1);
    JJv   = IIv;
    XXSv  = IIv;
    XXTv  = IIv;
    XXTv2 = IIv;
    IIvs  = zeros(Mesh.NELE * numv * nums, 1);
    JJvs  = IIvs;
    XXGvs = IIvs;
    iv    = 1;
    ivs   = 1;
end

for ie = 1:Mesh.NELE
    [gIs, gIv] = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
    [detJ, invJt] = CalcJacobian(Mesh.node(Mesh.ele(ie, :), :));

    S = zeros(nums);
    T = S;
    for iq = 1:length(wq2)
        S = S + detJ * ((invJt * dNs2{iq})' * ...
            (Mesh.epsr(Mesh.elab(ie)) * (invJt * dNs2{iq}))) * wq2(iq);
        T = T + detJ * (Ns2{iq}' * Ns2{iq}) * wq2(iq);
    end
    idx = is:is + nums^2 - 1;
    for j = 1:nums
        for k = 1:nums
            tmpIdx     = idx(nums * (j - 1) + k);
            IIs(tmpIdx)  = gIs(j);
            JJs(tmpIdx)  = gIs(k);
            XXSs(tmpIdx) = S(j, k);
            XXTs(tmpIdx) = T(j, k);
        end
    end
    is = is + nums^2;

    if vShap
        Sv  = zeros(numv);
        Tv  = Sv;
        Gvs = zeros(numv, nums);
        for iq = 1:length(wq2)
            x  = Ns2_1{iq};
            dx = invJt * dNs2_1{iq};
            switch Sys.pOrd
                case 1
                    Nv2(:, 1)  = x(2) * dx(:, 3) - x(3) * dx(:, 2);
                    Nv2(:, 2)  = x(1) * dx(:, 3) - x(3) * dx(:, 1);
                    Nv2(:, 3)  = x(1) * dx(:, 2) - x(2) * dx(:, 1);
                    dNv2(:, 1) = 2;
                    dNv2(:, 2) = -2;
                    dNv2(:, 3) = 2;
                case 2
                    Nv2(:, 1)  = x(2) * dx(:, 3) - x(3) * dx(:, 2);
                    Nv2(:, 2)  = x(1) * dx(:, 3) - x(3) * dx(:, 1);
                    Nv2(:, 3)  = x(1) * dx(:, 2) - x(2) * dx(:, 1);
                    Nv2(:, 4)  = 4 * (x(2) * dx(:, 3) + x(3) * dx(:, 2));
                    Nv2(:, 5)  = 4 * (x(1) * dx(:, 3) + x(3) * dx(:, 1));
                    Nv2(:, 6)  = 4 * (x(1) * dx(:, 2) + x(2) * dx(:, 1));
                    Nv2(:, 7)  = (x(1) * x(2) * dx(:, 3) - x(1) * x(3) * dx(:, 2));
                    Nv2(:, 8)  = (x(1) * x(2) * dx(:, 3) - x(2) * x(3) * dx(:, 1));
                    dNv2(:, 1) = 2;
                    dNv2(:, 2) = -2;
                    dNv2(:, 3) = 2;
                    dNv2(:, 4) = 0;
                    dNv2(:, 5) = 0;
                    dNv2(:, 6) = 0;
                    dNv2(:, 7) = 2 * x(1) - x(2) - x(3);
                    dNv2(:, 8) = x(1) + x(3) - 2 * x(2);
                case 3
                    Nv2(:, 1)  = x(2) * dx(:, 3) - x(3) * dx(:, 2);
                    Nv2(:, 2)  = x(1) * dx(:, 3) - x(3) * dx(:, 1);
                    Nv2(:, 3)  = x(1) * dx(:, 2) - x(2) * dx(:, 1);
                    Nv2(:, 4)  = 4 * (x(2) * dx(:, 3) + x(3) * dx(:, 2));
                    Nv2(:, 5)  = 4 * (x(1) * dx(:, 3) + x(3) * dx(:, 1));
                    Nv2(:, 6)  = 4 * (x(1) * dx(:, 2) + x(2) * dx(:, 1));
                    Nv2(:, 7)  = (x(1) * x(2) * dx(:, 3) - x(1) * x(3) * dx(:, 2));
                    Nv2(:, 8)  = (x(1) * x(2) * dx(:, 3) - x(2) * x(3) * dx(:, 1));
                    Nv2(:, 9)  = x(2) * (x(2) - 2 * x(3)) * dx(:, 3) ...
                                 + x(3) * (-x(3) + 2 * x(2)) * dx(:, 2);
                    Nv2(:, 10) = x(1) * (x(1) - 2 * x(3)) * dx(:, 3) ...
                                 + x(3) * (-x(3) + 2 * x(1)) * dx(:, 1);
                    Nv2(:, 11) = x(1) * (x(1) - 2 * x(2)) * dx(:, 2) ...
                                 + x(2) * (-x(2) + 2 * x(1)) * dx(:, 1);
                    Nv2(:, 12) = x(2) * x(1) * dx(:, 3) ...
                                 + x(1) * x(3) * dx(:, 2) ...
                                 + x(2) * x(3) * dx(:, 1);
                    Nv2(:, 13) = -x(2) * x(1) * (x(2) - 2 * x(3)) * dx(:, 3) ...
                                 - x(1) * x(3) * (-x(3) + 2 * x(2)) * dx(:, 2) ...
                                 + 3 * x(2) * x(3) * (x(2) - x(3)) * dx(:, 1);
                    Nv2(:, 14) = -x(1) * x(2) * (x(1) - 2 * x(3)) * dx(:, 3) ...
                                 + 3 * x(1) * x(3) * (x(1) - x(3)) * dx(:, 2) ...
                                 - x(2) * x(3) * (-x(3) + 2 * x(1)) * dx(:, 1);
                    Nv2(:, 15) = 3 * x(1) * x(2) * (x(1) - x(2)) * dx(:, 3) ...
                                 - x(1) * x(3) * (x(1) - 2 * x(2)) * dx(:, 2) ...
                                 - x(2) * x(3) * (-x(2) + 2 * x(1)) * dx(:, 1);
                    dNv2(:, 1)  = 2;
                    dNv2(:, 2)  = -2;
                    dNv2(:, 3)  = 2;
                    dNv2(:, 4)  = 0;
                    dNv2(:, 5)  = 0;
                    dNv2(:, 6)  = 0;
                    dNv2(:, 7)  = 2 * x(1) - x(2) - x(3);
                    dNv2(:, 8)  = x(1) + x(3) - 2 * x(2);
                    dNv2(:, 9)  = 0;
                    dNv2(:, 10) = 0;
                    dNv2(:, 11) = 0;
                    dNv2(:, 12) = 0;
                    dNv2(:, 13) = -16 * x(2) * x(3) + 4 * x(3)^2 + 4 * x(2)^2;
                    dNv2(:, 14) =  16 * x(1) * x(3) - 4 * x(3)^2 - 4 * x(1)^2;
                    dNv2(:, 15) = -16 * x(2) * x(1) + 4 * x(2)^2 + 4 * x(1)^2;
                otherwise
                    error('order must be between 1..3');
            end
            dNv2 = dNv2 * (dx(1, 2) * dx(2, 3) - dx(2, 2) * dx(1, 3));
            Sv   = Sv + detJ * (dNv2.' * dNv2) * wq2(iq);
            Tv   = Tv + detJ * (Nv2.' * Nv2) * wq2(iq);
            Gvs  = Gvs + detJ * (Nv2.' * (invJt * dNs2{iq})) * wq2(iq);
        end
        idxt = iv:iv + numv^2 - 1;
        for j = 1:numv
            for k = 1:numv
                tmpIdx      = idxt(numv * (j - 1) + k);
                IIv(tmpIdx)   = gIv(j);
                JJv(tmpIdx)   = gIv(k);
                XXSv(tmpIdx)  = Sv(j, k);
                XXTv(tmpIdx)  = Tv(j, k) * Mesh.epsr(Mesh.elab(ie));
                XXTv2(tmpIdx) = Tv(j, k);
            end
        end
        iv = iv + numv^2;

        idxg = ivs:ivs + numv * nums - 1;
        for j = 1:numv
            for k = 1:nums
                tmpIdx       = idxg(nums * (j - 1) + k);
                IIvs(tmpIdx) = gIv(j);
                JJvs(tmpIdx) = gIs(k);
                XXGvs(tmpIdx)= Gvs(j, k);
            end
        end
        ivs = ivs + numv * nums;
    end
end

%% Store sparse system matrices
Sys.S  = sparse(IIs, JJs, XXSs, Sys.NDOFs, Sys.NDOFs);
Sys.T  = sparse(IIs, JJs, XXTs, Sys.NDOFs, Sys.NDOFs);
Sys.fs = sparse(Sys.NDOFs, 1);

if vShap
    Sys.St  = sparse(IIv, JJv, XXSv, Sys.NDOFv, Sys.NDOFv);
    Sys.Tt  = sparse(IIv, JJv, XXTv, Sys.NDOFv, Sys.NDOFv);
    Sys.Tt2 = sparse(IIv, JJv, XXTv2, Sys.NDOFv, Sys.NDOFv);
    Sys.G   = sparse(IIvs, JJvs, XXGvs, Sys.NDOFv, Sys.NDOFs);
end

%% Boundary conditions evaluation
%% -- ABC (absorbing boundary condition)
if FlagABC
    Sys.fsEinc = exp(-1i * Sys.k ...
        * (Sys.kEinc * Mesh.refNode.').');  % scattered field formulation
    NSABC  = length(idsABC);
    IIABC  = ones(NSABC * (Sys.pOrd + 1)^2, 1);
    JJABC  = IIABC;
    XXABC  = IIABC - 1;
    is     = 1;
    for ie = 1:Mesh.NELE
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
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnABC(i)).';
                l   = diff(Mesh.node(nodeId, :));
                n   = cross(cross([l 0], ...
                    [diff(Mesh.node([nodeId(1) intNode], :)) 0]), [l 0]);
                n = n / norm(n);  % inward normal

                TrBC = zeros(Sys.pOrd + 1);
                for iq = 1:length(wq1)
                    TrBC = TrBC + norm(l) * (Ns1{iq}' * Ns1{iq}) * wq1(iq);
                end

                rho = [ones(length(wq1), 1) * Mesh.node(nodeId(1), :) ...
                       + xq1 * l, zeros(length(wq1), 1)];
                v   = [0 0 1];
                Inc = dot(v, (v - cross(n, cross(Sys.kEinc, v)))) ...
                    * exp(-1i * Sys.k ...
                    * dot(Sys.kEinc.' * ones(1, Sys.pOrd + 1), rho.'));
                frBC = zeros(Sys.pOrd + 1, 1);
                for iq = 1:length(wq1)
                    frBC = frBC + norm(l) ...
                        * (Ns1{iq} .* (Inc(:, iq))).' * wq1(iq);
                end

                idx = is:is + (Sys.pOrd + 1)^2 - 1;
                for j = 1:(Sys.pOrd + 1)
                    for k = 1:(Sys.pOrd + 1)
                        tmpIdx       = idx((Sys.pOrd + 1) * (j - 1) + k);
                        IIABC(tmpIdx) = gIs(j);
                        JJABC(tmpIdx) = gIs(k);
                        XXABC(tmpIdx) = TrBC(j, k);
                    end
                end
                is = is + (Sys.pOrd + 1)^2;
                Sys.fs(gIs) = Sys.fs(gIs) + frBC;  % total field formulation
            end
        end
    end
    Sys.ABC    = sparse(IIABC, JJABC, XXABC, Sys.NDOFs, Sys.NDOFs);
    Sys.DirABC = unique(IIABC(IIABC > 0));
end

%% -- DD (domain-decomposition interface)
if FlagDD
    NSDD  = length(idsDD);
    IIDD  = ones(NSDD * (Sys.pOrd + 1)^2, 1);
    JJDD  = IIDD;
    XXDD  = IIDD * 0;
    is    = 1;
    for ie = 1:Mesh.NELE
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
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDD(i)).';
                l   = diff(Mesh.node(nodeId, :));
                n   = cross([l 0], cross([l 0], ...
                    [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                n = n / norm(n);

                TrBC = zeros(Sys.pOrd + 1);
                for iq = 1:length(wq1)
                    TrBC = TrBC + norm(l) * (Ns1{iq}' * Ns1{iq}) * wq1(iq);
                end

                rho = [ones(length(wq1), 1) * Mesh.node(nodeId(1), :) ...
                       + xq1 * l, zeros(length(wq1), 1)];
                dEinc = -1i * Sys.k * dot(Sys.kEinc, n) ...
                    * exp(-1i * Sys.k ...
                    * dot(Sys.kEinc.' * ones(1, Sys.pOrd + 1), rho.'));
                frBC = zeros(Sys.pOrd + 1, 1);
                for iq = 1:length(wq1)
                    frBC = frBC + norm(l) * (Ns1{iq} .* dEinc).' * wq1(iq);
                end

                idx = is:is + (Sys.pOrd + 1)^2 - 1;
                for j = 1:(Sys.pOrd + 1)
                    for k = 1:(Sys.pOrd + 1)
                        tmpIdx      = idx((Sys.pOrd + 1) * (j - 1) + k);
                        IIDD(tmpIdx) = gIs(j);
                        JJDD(tmpIdx) = gIs(k);
                        XXDD(tmpIdx) = TrBC(j, k);
                    end
                end
                is = is + (Sys.pOrd + 1)^2;
                Sys.fs(gIs) = Sys.fs(gIs) + frBC;
            end
        end
    end
    Sys.DD    = sparse(IIDD, JJDD, XXDD, Sys.NDOFs, Sys.NDOFs);
    Sys.DirDD = unique(IIDD(IIDD > 0));
end

%% -- Dirichlet boundary condition
if FlagDir
    for ibc = 1:length(idsDir)
        NSDir  = length(idsDir{ibc});
        IIDir  = zeros(NSDir * (Sys.pOrd + 1)^2, 1);
        is     = 1;
        for ie = 1:Mesh.NELE
            onDir = [sum(abs(Mesh.spig(ie, 1)) == idsDir{ibc}) ...
                     sum(abs(Mesh.spig(ie, 2)) == idsDir{ibc}) ...
                     sum(abs(Mesh.spig(ie, 3)) == idsDir{ibc})];
            if sum(onDir) > 0
                idOnDir = find(onDir == 1);
                for i = 1:length(idOnDir)
                    gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDir(i));
                    spigId  = abs(Mesh.spig(ie, idOnDir(i)));
                    nodeId  = Mesh.spig2(spigId, :);
                    intNode = Mesh.ele(ie, ...
                        Mesh.ele(ie, :) ~= nodeId(1) ...
                        & Mesh.ele(ie, :) ~= nodeId(2));
                    l = diff(Mesh.node(nodeId, :));
                    n = cross([l 0], cross([l 0], ...
                        [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                    n = n / norm(n);
                    frBC = zeros(Sys.pOrd + 1, 1);
                    for iq = 1:length(wq1)
                        frBC = frBC + norm(l) * (Ns1{iq}).' * wq1(iq);
                    end
                    Sys.fs(gIs) = Sys.fs(gIs) + frBC;
                    idx = is:is + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx      = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIDir(tmpIdx) = gIs(j);
                        end
                    end
                    is = is + (Sys.pOrd + 1)^2;
                end
            end
        end
        Sys.Dir{ibc} = unique(IIDir(IIDir > 0));
    end
end

%% -- Neumann boundary condition
if FlagNeu
    NSNeu  = length(idsNeu);
    IINeu  = zeros(NSNeu * (Sys.pOrd + 1)^2, 1);
    is     = 1;
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
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnNeu(i));
                l   = diff(Mesh.node(nodeId, :));
                n   = cross([l 0], cross([l 0], ...
                    [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                n = n / norm(n);
                frBC = zeros(Sys.pOrd + 1, 1);
                for iq = 1:length(wq1)
                    frBC = frBC + norm(l) * (Ns1{iq}).' * wq1(iq);
                end
                idx = is:is + (Sys.pOrd + 1)^2 - 1;
                for j = 1:(Sys.pOrd + 1)
                    for k = 1:(Sys.pOrd + 1)
                        tmpIdx       = idx((Sys.pOrd + 1) * (j - 1) + k);
                        IINeu(tmpIdx) = gIs(j);
                    end
                end
                is = is + (Sys.pOrd + 1)^2;
                Sys.fs(gIs) = Sys.fs(gIs) + frBC;
            end
        end
    end
    Sys.Neu = unique(IINeu(IINeu > 0));
end

%% -- Waveguide port (WP) eigenvalue extraction
if FlagWP
    for ibc = 1:length(idsWP)
        NSWP  = length(idsWP{ibc});
        IIsWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        JJsWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        SSsWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        TTsWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        is    = 1;
        for ie = 1:Mesh.NELE
            onDir = [sum(abs(Mesh.spig(ie, 1)) == idsWP{ibc}) ...
                     sum(abs(Mesh.spig(ie, 2)) == idsWP{ibc}) ...
                     sum(abs(Mesh.spig(ie, 3)) == idsWP{ibc})];
            if sum(onDir) > 0
                idOnDir = find(onDir == 1);
                Stt = zeros((Sys.pOrd + 1), (Sys.pOrd + 1));
                Ttt = zeros((Sys.pOrd + 1), (Sys.pOrd + 1));
                for i = 1:length(idOnDir)
                    gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDir(i));
                    spigId  = abs(Mesh.spig(ie, idOnDir(i)));
                    nodeId  = Mesh.spig2(spigId, :);
                    intNode = Mesh.ele(ie, ...
                        Mesh.ele(ie, :) ~= nodeId(1) ...
                        & Mesh.ele(ie, :) ~= nodeId(2));
                    l    = diff(Mesh.node(nodeId, :));
                    n    = cross([l 0], cross([l 0], ...
                        [diff(Mesh.node([nodeId(1) intNode], :)) 0]));
                    n    = n / norm(n);
                    detJ = norm(l);
                    for iq = 1:length(wq1)
                        Stt = Stt + norm(l) ...
                            * ((dNs1{iq} / detJ).' * (dNs1{iq} / detJ)) * wq1(iq);
                        Ttt = Ttt + norm(l) ...
                            * (Ns1{iq}.' * Ns1{iq}) * wq1(iq);
                    end
                    idx = is:is + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx       = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIsWP(tmpIdx) = gIs(j);
                            JJsWP(tmpIdx) = gIs(k);
                            SSsWP(tmpIdx) = Stt(j, k);
                            TTsWP(tmpIdx) = Ttt(j, k);
                        end
                    end
                    is = is + (Sys.pOrd + 1)^2;
                end
            end
        end
        SspWP       = sparse(IIsWP, JJsWP, SSsWP, Sys.NDOFs, Sys.NDOFs);
        TspWP       = sparse(IIsWP, JJsWP, TTsWP, Sys.NDOFs, Sys.NDOFs);
        Sys.WP{ibc} = unique(IIsWP(IIsWP > 0));
        NDoFsWP     = length(Sys.WP{ibc});
        for i = 1:NDoFsWP
            for j = 1:length(Sys.Dir)
                if find(Sys.WP{ibc}(NDoFsWP - i + 1) == Sys.Dir{j})
                    Sys.WP{ibc}(NDoFsWP - i + 1) = [];
                end
            end
        end
        SspWP = SspWP(Sys.WP{ibc}, Sys.WP{ibc});
        TspWP = TspWP(Sys.WP{ibc}, Sys.WP{ibc});
        [evect, eval] = eigs(SspWP, TspWP, Sys.WPnModes, 'SM');
        [eval, idx]   = sort(diag(eval), 'ascend');
        evect         = evect(:, idx);
        Sys.WPvec{ibc} = evect / sqrt(evect.' * TspWP * evect);
        Sys.WPfc{ibc}  = sqrt(eval).' * Sys.c0 / 2 / pi;
    end
end

Sys.bypass = true;

end
