function [Sys, Mesh] = AssembHB(Sys, Mesh)
% Assemble harmonic-balance (multi-frequency) FE system matrices S and T
% with optional Kerr nonlinearity coupling.

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

%% Master matrix dimensions and quadrature
[nums, numv] = CalcOrderMatSize(Sys.pOrd);
[xq1, wq1]   = CalcSimplexQuad(Sys.pOrd + 1, 1);
[xyq2, wq2]  = CalcSimplexQuad(Sys.pOrd + 1, 2);

%% Shape functions
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

%% Harmonics setup
nHarms  = length(Sys.HBharms);
f1      = Sys.freq * Sys.HBharms(1);
f2      = Sys.freq * Sys.HBharms(2);
df      = min([f1 f2 abs(round(f1 - f2))]);
bf      = gcd(df, f1);
N       = ceil(max(Sys.HBharms) * 2 * f1 / bf);
N       = N + mod(N + 1, 2);
t       = linspace(0, 1 / bf * (N - 1) / N, N);
fnl     = bf * [0 1:floor(N / 2) -floor(N / 2):-1];

Cx   = cos(2 * pi * fnl.' * t);
Sx   = sin(2 * pi * Sys.freq * Sys.HBharms.' * t);
posL = uint32(Sys.freq * Sys.HBharms / bf) + 1;
posR = N - uint32(Sys.freq * Sys.HBharms / bf) + 1;

%% Assembly -- element loop with harmonic coupling
II  = ones(Mesh.NELE * nums^2 * nHarms^2, 1);
JJ  = II;
XXS = II * 0;
XXT = II * 0;
s   = 1;

for ie = 1:Mesh.NELE
    if ~isfield(Sys, 'u')
        Sys.u = zeros(Sys.NDOFs, 1);
    end
    [gIs] = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
    [detJ, invJt] = CalcJacobian(Mesh.node(Mesh.ele(ie, :), :));

    %% Kerr nonlinearity coupling matrix
    if isfield(Mesh, 'kerr')
        E = zeros(nHarms, 1);
        for jh = 1:nHarms
            sol   = Sys.u0(gIs + (jh - 1) * Sys.NDOFs);
            E(jh) = Shape2(.5, .5) * sol;
        end
        if isfield(Sys, 'FFT')
            [EpsrC] = GetCouplKerrFFT(nHarms, abs(E), ...
                Mesh.epsr(Mesh.elab(ie)), Mesh.kerr(Mesh.elab(ie)), ...
                Cx, Sx, posR, posL);
        else
            [EpsrC] = GetCouplKerrAnalyt(nHarms, abs(E), ...
                Mesh.epsr(Mesh.elab(ie)), Mesh.kerr(Mesh.elab(ie)));
        end
        NurC = eye(nHarms);
    else
        NurC  = eye(nHarms);
        EpsrC = Mesh.epsr(Mesh.elab(ie)) * eye(nHarms);
    end

    %% Element stiffness and mass
    S  = zeros(nums);
    T  = S;
    nu = Mesh.mur{Mesh.elab(ie)} \ eye(2);
    for iq = 1:length(wq2)
        S = S + detJ * ((invJt * dNQuad2{iq})' * nu ...
            * (invJt * dNQuad2{iq})) * wq2(iq);
        T = T + detJ * (NQuad2{iq}' * NQuad2{iq}) * wq2(iq);
    end

    %% Assemble harmonic-block matrix
    idx = s:s + nums^2 * nHarms^2 - 1;
    for jh = 1:nHarms
        for kh = 1:nHarms
            for j = 1:nums
                for k = 1:nums
                    tmpIdx = idx(((jh - 1) * nHarms + kh - 1) * nums^2 ...
                                + nums * (j - 1) + k);
                    II(tmpIdx)  = gIs(j) + Sys.NDOFs * (jh - 1);
                    JJ(tmpIdx)  = gIs(k) + Sys.NDOFs * (kh - 1);
                    XXS(tmpIdx) = S(j, k) * NurC(jh, kh);
                    XXT(tmpIdx) = Sys.HBharms(jh)^2 * T(j, k) * EpsrC(jh, kh);
                end
            end
        end
    end
    s = s + nums^2 * nHarms^2;
end

Sys.S = sparse(II, JJ, XXS, Sys.NDOFs * nHarms, Sys.NDOFs * nHarms);
Sys.T = sparse(II, JJ, XXT, Sys.NDOFs * nHarms, Sys.NDOFs * nHarms);
Sys.f = sparse(Sys.NDOFs * nHarms, 1);

%% Boundary conditions
%% -- ABC (absorbing boundary condition)
if FlagABC
    Sys.fEinc = exp(-1i * Sys.k ...
        * (Sys.kEinc * Mesh.refNode.').');  % scattered field formulation

    NSABC  = length(idsABC);
    IIABC  = ones(NSABC * (Sys.pOrd + 1)^2, 1);
    JJABC  = IIABC;
    XXABC  = IIABC * 0;
    s      = 1;
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
                    TrBC = TrBC + norm(l) * (NQuad1{iq}' * NQuad1{iq}) * wq1(iq);
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
                        * (NQuad1{iq} .* (Inc(:, iq))).' * wq1(iq);
                end

                idx = s:s + (Sys.pOrd + 1)^2 - 1;
                for j = 1:(Sys.pOrd + 1)
                    for k = 1:(Sys.pOrd + 1)
                        tmpIdx       = idx((Sys.pOrd + 1) * (j - 1) + k);
                        IIABC(tmpIdx) = gIs(j);
                        JJABC(tmpIdx) = gIs(k);
                        XXABC(tmpIdx) = TrBC(j, k);
                    end
                end
                s = s + (Sys.pOrd + 1)^2;
                Sys.f(gIs) = Sys.f(gIs) + frBC;  % total field formulation
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
    s     = 1;
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
                        IIDD(tmpIdx) = gIs(j);
                        JJDD(tmpIdx) = gIs(k);
                        XXDD(tmpIdx) = TrBC(j, k);
                    end
                end
                s = s + (Sys.pOrd + 1)^2;
                Sys.f(gIs) = Sys.f(gIs) + frBC;
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
        s      = 1;
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
                        frBC = frBC + norm(l) * (NQuad1{iq}).' * wq1(iq);
                    end
                    Sys.f(gIs) = Sys.f(gIs) + frBC;
                    idx = s:s + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx      = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIDir(tmpIdx) = gIs(j);
                        end
                    end
                    s = s + (Sys.pOrd + 1)^2;
                end
            end
        end
        Sys.Dir{ibc} = unique(IIDir(IIDir > 0));
        DirIds = Sys.Dir{ibc};
        for jh = 2:nHarms
            DirIds = [DirIds; Sys.Dir{ibc} + Sys.NDOFs * (jh - 1)];
        end
        Sys.Dir{ibc} = DirIds;
    end
end

%% -- Neumann boundary condition
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
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnNeu(i));
                l   = diff(Mesh.node(nodeId, :));
                n   = cross([l 0], cross([l 0], ...
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
                        IINeu(tmpIdx) = gIs(j);
                    end
                end
                s = s + (Sys.pOrd + 1)^2;
                Sys.f(gIs) = Sys.f(gIs) + frBC;
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
        IIvWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        JJvWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        SSvWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        TTvWP = zeros(NSWP * (Sys.pOrd + 1)^2, 1);
        s     = 1;
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
                            * ((dNQuad1{iq} / detJ).' ...
                               * (dNQuad1{iq} / detJ)) * wq1(iq);
                        Ttt = Ttt + norm(l) ...
                            * (NQuad1{iq}.' * NQuad1{iq}) * wq1(iq);
                    end
                    idx = s:s + (Sys.pOrd + 1)^2 - 1;
                    for j = 1:(Sys.pOrd + 1)
                        for k = 1:(Sys.pOrd + 1)
                            tmpIdx       = idx((Sys.pOrd + 1) * (j - 1) + k);
                            IIsWP(tmpIdx) = gIs(j);
                            JJsWP(tmpIdx) = gIs(k);
                            SSsWP(tmpIdx) = Stt(j, k);
                            TTsWP(tmpIdx) = Ttt(j, k);
                        end
                    end
                    s = s + (Sys.pOrd + 1)^2;
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
