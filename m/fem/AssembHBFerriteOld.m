function [Sys, Mesh] = AssembHBFerrite(Sys, Mesh)
FlagABC = false;
FlagDir = false;
FlagNeu = false;
FlagDD = false;
FlagWP = false;
if ~isfield(Sys,'bypass')
    %% constants
    Sys = GetConstants(Sys);
    %% raw DoF
    Sys = CalcDoFsNumber(Sys, Mesh);
    Mesh = CalcDoFsPositions(Sys, Mesh);
    if isfield(Mesh,'BC')
        if isfield(Mesh.BC,'ABC')
            idsABC = find(Mesh.slab == Mesh.BC.ABC);
            Sys.idsABC = idsABC;
            FlagABC = true;
        end
        if isfield(Mesh.BC,'Dir')
            idsDir = cell(length(Mesh.BC.Dir),1);
            for ibc=1:length(Mesh.BC.Dir)
                idsDir{ibc} = find(Mesh.slab == Mesh.BC.Dir(ibc));
            end
            FlagDir = true;
        end
        if isfield(Mesh.BC,'Neu')
            idsNeu = find(Mesh.slab == Mesh.BC.Neu);
            FlagNeu = true;
        end
        if isfield(Mesh.BC,'DD')
            idsDD = find(Mesh.slab == Mesh.BC.DD);
            Sys.idsDD = idsDD;
            FlagDD = true;
        end
        if isfield(Mesh.BC,'WP')
            idsWP = cell(length(Mesh.BC.WP),1);
            for ibc=1:length(Mesh.BC.WP)
                idsWP{ibc} = find(Mesh.slab == Mesh.BC.WP(ibc));
            end
            FlagWP = true;
        end
    end
    nMtrl = length(unique(Mesh.elab));
    if ~isfield(Mesh,'epsr')
        Mesh.epsr = ones(1,nMtrl);
    end
    if ~isfield(Mesh,'mur')
        Mesh.mur = cell(1,nMtrl);
        for i=1:nMtrl
            Mesh.mur{i} = eye(2);
        end
    end
end

[nums, numv] = CalcOrderMatSize(Sys.pOrd);
[xq1, wq1] = CalcSimplexQuad(Sys.pOrd+1,1);
[xyq2, wq2] = CalcSimplexQuad(Sys.pOrd+1,2);
[Shape1, Shape1Deriv] = CalcShapeFunctions(1, Sys.pOrd);
[Shape2, Shape2DerivX, Shape2DerivY] = CalcShapeFunctions(2, Sys.pOrd);
% [Shape2_1, Shape2DerivX_1, Shape2DerivY_1] = CalcShapeFunctions(2, 1);
NQuad1 = cell(1,length(wq1));
dNQuad1 = cell(1,length(wq1));
for iq=1:length(wq1)
    NQuad1{iq} = Shape1(xq1(iq));
    dNQuad1{iq} = Shape1Deriv(xq1(iq));
end
NQuad2 = cell(1,length(wq2));
dNQuad2 = cell(1,length(wq2));
for iq=1:length(wq2)
    NQuad2{iq} = Shape2(xyq2(iq,1),xyq2(iq,2));
    dNQuad2{iq} = [Shape2DerivX(xyq2(iq,1),xyq2(iq,2));...
        Shape2DerivY(xyq2(iq,1),xyq2(iq,2))];
end
%% harmonics
nHarms = Sys.nHarms;
Ferr = Mesh.Ferr;
omega = 2*pi*Sys.freq;
Mtrl = GetMtrlParamsFFT(Sys);
% disp(Mtrl.bf);
Field.Sin = sin(2*pi*Sys.freq*Sys.HBharms.'*Mtrl.t);
Field.Cos = cos(2*pi*Sys.freq*Sys.HBharms.'*Mtrl.t);
if isfield(Sys,'SinOnly')
    Sys.Harms = Sys.HBharms;
    Mtrl.Harms = zeros(1,Mtrl.nHarms*2);
    Mtrl.Harms(1:2:end) = Sys.HBharms;
    Mtrl.Harms(2:2:end) = Sys.HBharms;
else
    Sys.Harms = zeros(nHarms,1);
    Sys.Harms(1:2:end) = Sys.HBharms;
    Sys.Harms(2:2:end) = Sys.HBharms;
    Mtrl.Harms = Sys.Harms;
end
Mtrl.Sin = sin(2*pi*Mtrl.fnl.'*Mtrl.t);
Mtrl.Cos = cos(2*pi*Mtrl.fnl.'*Mtrl.t);
Hconst = 1/(1i*omega*Sys.mu0);
MtrlMur = Mtrl;
MtrlKr = Mtrl;
val = Ferr.Gamma * Ferr.alpha;
MtrlMur.func = @(x,y,z) (1 + (Ferr.w0+1i.*Ferr.aDH).*Ferr.wm./ ...
    ((Ferr.w0+1i.*Ferr.aDH).^2 + sum(((val.*x*y)).^2,1) - (z.*omega).^2) );
MtrlKr.func = @(x,y,z) ((z.*omega).*Ferr.wm./((Ferr.w0+1i.*Ferr.aDH).^2 ...
    + sum(((val.*x*y)).^2,1) - (z*omega).^2) );
%% assemb
II = ones(Mesh.NELE*nums^2*nHarms^2,1);
JJ = II;
XXS = II*0;
XXT = II*0;
s=1;
% find(Mesh.elab == Mesh.NLlab)
% ie = 9;
for ie=1:Mesh.NELE
%     fprintf('+');
    if ~isfield(Sys,'u')
        Sys.u = zeros(nHarms*Sys.NDOFs,1);
    end
    [gIs] = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
    [detJ, invJt] = CalcJacobian(Mesh.node(Mesh.ele(ie,:),:));
    NurC = cell(nHarms,nHarms);
    if Mesh.elab(ie) == Mesh.NLlab
        if isempty(Mesh.NurC{ie})
            for jh=1:nHarms
                mur = 1 + (Ferr.w0+1i.*Ferr.aDH).*Ferr.wm./...
                    ((Ferr.w0+1i.*Ferr.aDH).^2-(Sys.Harms(jh)*omega).^2);
                kr = omega.*Ferr.wm./((Ferr.w0+1i.*Ferr.aDH).^2-(Sys.Harms(jh)*omega).^2);
                for kh = 1:nHarms
                    Mesh.NurC{ie}{jh,kh} = [mur 1i*kr; -1i*kr mur]\eye(2)* (jh==kh);
                end
            end
        end

        if isfield(Sys,'SinOnly')
            Field.E = zeros(2*nHarms,1);
            Field.H = zeros(2*nHarms,2);
            for jh=1:nHarms
                sol = Sys.u0(gIs+(jh-1)*Sys.NDOFs);
                Field.E(2*jh-1,1) = Field.E(2*jh-1,1) + ...
                    Shape2(.5,.5)*sol;
                for kh=jh:nHarms
                    sol = Sys.u0(gIs+(kh-1)*Sys.NDOFs);
                    dE = invJt*[Shape2DerivX(.5,.5); Shape2DerivY(.5,.5)]*sol;
                    Field.H(2*jh-1,:) = Field.H(2*jh-1,:) + ...
                        Sys.Harms(kh)\Hconst*(Mesh.NurC{ie}{jh,kh}*...
                        [dE(2); -dE(1)]).';
                end
            end
        else
            Field.E = zeros(nHarms,1);
            Field.H = zeros(nHarms,2);
            for jh=1:nHarms
                sol = Sys.u0(gIs+(jh-1)*Sys.NDOFs);
                Field.E(jh,1) = Field.E(jh,1) + ...
                    Shape2(.5,.5)*sol;
                for kh=1:nHarms
                    sol = Sys.u0(gIs+(kh-1)*Sys.NDOFs);
                    Field.H(jh,:) = Field.H(jh,:) + ...
                        (Sys.Harms(kh)\Hconst*Mesh.NurC{ie}{jh,kh}*...
                        invJt*[Shape2DerivY(.5,.5); -Shape2DerivX(.5,.5)]*sol).';
                end
            end
        end
        if ie==9
%             format short
%             disp(sum(abs(Field.H(1:2:end,:)).^2,2))
            disp(abs(Field.H(1:2:end,:)))
            disp(Mesh.NurC{ie}{1,3})
            disp(Mesh.NurC{ie}{1,4})
% %             Field.E
        end
        [MtrlMur] = GetHCouplFFT(MtrlMur, Field);
        [MtrlKr] = GetHCouplFFT(MtrlKr, Field);

%         Mtrl.func = @(x,y) (1 + (Ferr.w0+1i.*Ferr.aDH).*Ferr.wm ./ ...
%             ((Ferr.w0+1i.*Ferr.aDH).^2 - (y*omega).^2))* ( 1 + Ferr.alpha*sum(abs(x).^2,1) );
%         [MtrlMur] = GetHCouplFFT(Mtrl, Field);
%         Mtrl.func = @(x,y) ((y*omega).*Ferr.wm ./ ((Ferr.w0+1i.*Ferr.aDH).^2 ...
%             - (y*omega).^2)) * ( 1 + Ferr.alpha*sum(abs(x).^2,1) );
%         [MtrlKr] = GetHCouplFFT(Mtrl, Field);

        if isfield(Sys,'SinOnly')
            MurC = MtrlMur.D(1:2:2*nHarms,1:2:2*nHarms);
            KrC = MtrlKr.D(1:2:2*nHarms,1:2:2*nHarms);
        else
            MurC = MtrlMur.D;
            KrC = MtrlKr.D;
        end

%         if ie==9
% %             format long;
%             disp(MurC)
%             disp(KrC)
%         end

        for jh=1:nHarms
            for kh = 1:nHarms
                lMuR = [MurC(jh,kh) 1i*KrC(jh,kh); -1i*KrC(jh,kh) MurC(jh,kh)];
                if abs(det(lMuR)) > eps
                    NurC{jh,kh} = lMuR\eye(2);
                else
                    NurC{jh,kh} = zeros(2);
                end
            end
        end
        EpsrC = Mesh.epsr(Mesh.elab(ie))*eye(nHarms);
    else
        % not on ferrite
        lmur = Mesh.mur(Mesh.elab(ie));
        lkr = Mesh.kr(Mesh.elab(ie));
        for jh=1:nHarms
            for kh = 1:nHarms
                NurC{jh,kh} = [lmur 1i*lkr; -1i*lkr lmur]\eye(2) * (jh==kh);
            end
        end
        EpsrC = Mesh.epsr(Mesh.elab(ie))*eye(nHarms);
    end
    % saving NurC for next field evaluation
    Mesh.NurC{ie} = NurC;
    S = cell(nHarms,nHarms);
    T = cell(nHarms,nHarms);
    for jh=1:nHarms
        for kh = 1:nHarms
            S{jh,kh} = zeros(nums);
            T{jh,kh} = zeros(nums);
            for iq=1:length(wq2)
                S{jh,kh} = S{jh,kh} + detJ*((invJt*dNQuad2{iq})'*...
                    NurC{jh,kh}*(invJt*dNQuad2{iq}))*wq2(iq);
                T{jh,kh} = T{jh,kh} + ...
                    detJ*(NQuad2{iq}'*EpsrC(jh,kh)*NQuad2{iq})*wq2(iq);
            end
        end
    end

    idx = s:s+nums^2*nHarms^2-1;
    for jh=1:nHarms
        for kh = 1:nHarms
            for j=1:nums
                for k = 1:nums
                    tmpIdx = idx(((jh-1)*nHarms + kh-1)*nums^2+nums*(j-1)+k);
                    II(tmpIdx) = gIs(j)+Sys.NDOFs*(jh-1);
                    JJ(tmpIdx) = gIs(k)+Sys.NDOFs*(kh-1);
                    XXS(tmpIdx) = S{jh,kh}(j,k);
                    XXT(tmpIdx) = Sys.Harms(jh)^2*T{jh,kh}(j,k);
                end
            end
        end
    end
    s = s+nums^2*nHarms^2;
end

Sys.S = sparse(II,JJ,XXS,Sys.NDOFs*nHarms,Sys.NDOFs*nHarms);
Sys.T = sparse(II,JJ,XXT,Sys.NDOFs*nHarms,Sys.NDOFs*nHarms);
Sys.f = sparse(Sys.NDOFs*nHarms,1);

if FlagABC
    Sys.fEinc = exp(-1i * Sys.k * (Sys.kEinc * Mesh.refNode.').'); % for scattered field formulation

    NSABC = length(idsABC);
    IIABC = ones(NSABC*(Sys.pOrd+1)^2,1);
    JJABC = IIABC;
    XXABC = IIABC*0;
    s=1;
    for ie=1:Mesh.NELE
        onABC = [sum(abs(Mesh.spig(ie,1)) == idsABC) ...
            sum(abs(Mesh.spig(ie,2)) == idsABC) ...
            sum(abs(Mesh.spig(ie,3)) == idsABC)];
        if sum(onABC) > 0
            idOnABC = find(onABC==1);
            for i=1:length(idOnABC)
                spigId = Mesh.spig(ie,idOnABC(i));
                nodeId = Mesh.spig2(abs(spigId),:);
                intNode = Mesh.ele(ie, ...
                    Mesh.ele(ie,:)~=nodeId(1) & Mesh.ele(ie,:)~=nodeId(2));
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnABC(i)).';
                l = diff(Mesh.node(nodeId,:));
                n = cross(cross([l 0],...
                    [diff(Mesh.node([nodeId(1) intNode] ,:)) 0]),[l 0]);
                n = n/norm(n); % normale entrante

                TrBC = zeros(Sys.pOrd+1);
                for iq=1:length(wq1)
                    TrBC = TrBC + norm(l)*(NQuad1{iq}'*NQuad1{iq})*wq1(iq);
                end

                rho = [ones(length(wq1),1)*Mesh.node(nodeId(1),:) + xq1*l, ...
                    zeros(length(wq1),1)];
                v = [0 0 1];
                Inc = dot(v, (v - cross(n,cross(Sys.kEinc,v)))) * ...
                    exp(-1i * Sys.k * dot(Sys.kEinc.'*ones(1,Sys.pOrd+1), rho.'));
                frBC = zeros(Sys.pOrd+1,1);
                for iq=1:length(wq1)
                    frBC = frBC + norm(l)*(NQuad1{iq}.*(Inc(:,iq))).'*wq1(iq);
                end

                idx = s:s+(Sys.pOrd+1)^2-1;
                for j=1:(Sys.pOrd+1)
                    for k = 1:(Sys.pOrd+1)
                        tmpIdx = idx((Sys.pOrd+1)*(j-1)+k);
                        IIABC(tmpIdx) = gIs(j);
                        JJABC(tmpIdx) = gIs(k);
                        XXABC(tmpIdx) = TrBC(j,k);
                    end
                end
                s = s+(Sys.pOrd+1).^2;
                Sys.f(gIs) = Sys.f(gIs) + frBC; % for total field formulation
            end
        end
    end
    Sys.ABC = sparse(IIABC,JJABC,XXABC,Sys.NDOFs,Sys.NDOFs);
    Sys.DirABC = unique(IIABC(IIABC>0));
end

if FlagDD
    NSDD = length(idsDD);
    IIDD = ones(NSDD*(Sys.pOrd+1)^2,1);
    JJDD = IIDD;
    XXDD = IIDD*0;
    s=1;
    for ie=1:Mesh.NELE
        onDD = [sum(abs(Mesh.spig(ie,1)) == idsDD) ...
            sum(abs(Mesh.spig(ie,2)) == idsDD) ...
            sum(abs(Mesh.spig(ie,3)) == idsDD)];
        if sum(onDD) > 0
            idOnDD = find(onDD==1);
            for i=1:length(idOnDD)
                spigId = Mesh.spig(ie,idOnDD(i));
                nodeId = Mesh.spig2(abs(spigId),:);
                intNode = Mesh.ele(ie, ...
                    Mesh.ele(ie,:)~=nodeId(1) & Mesh.ele(ie,:)~=nodeId(2));
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDD(i)).';
                l = diff(Mesh.node(nodeId,:));
                n = cross([l 0],cross([l 0],...
                    [diff(Mesh.node([nodeId(1) intNode] ,:)) 0]));
                n = n/norm(n);

                TrBC = zeros(Sys.pOrd+1);
                for iq=1:length(wq1)
                    TrBC = TrBC + norm(l)*(NQuad1{iq}'*NQuad1{iq})*wq1(iq);
                end

                rho = [ones(length(wq1),1)*Mesh.node(nodeId(1),:) + xq1*l, ...
                    zeros(length(wq1),1)];
                dEinc =  - 1i .* Sys.k .* dot(Sys.kEinc, n) .* ...
                    exp(-1i * Sys.k * dot(Sys.kEinc.'*ones(1,Sys.pOrd+1), rho.'));
                frBC = zeros(Sys.pOrd+1,1);
                for iq=1:length(wq1)
                    frBC = frBC + norm(l)*(NQuad1{iq}.*dEinc).'*wq1(iq);
                end

                idx = s:s+(Sys.pOrd+1)^2-1;
                for j=1:(Sys.pOrd+1)
                    for k = 1:(Sys.pOrd+1)
                        tmpIdx = idx((Sys.pOrd+1)*(j-1)+k);
                        IIDD(tmpIdx) = gIs(j);
                        JJDD(tmpIdx) = gIs(k);
                        XXDD(tmpIdx) = TrBC(j,k);
                    end
                end
                s = s+(Sys.pOrd+1).^2;
                Sys.f(gIs) = Sys.f(gIs) + frBC;
            end
        end
    end
    Sys.DD = sparse(IIDD,JJDD,XXDD,Sys.NDOFs,Sys.NDOFs);
    Sys.DirDD = unique(IIDD(IIDD>0));
end

if FlagDir
    for ibc=1:length(idsDir)
        NSDir = length(idsDir{ibc});
        IIDir = zeros(NSDir*(Sys.pOrd+1)^2,1);
        s=1;
        for ie=1:Mesh.NELE
            onDir = [sum(abs(Mesh.spig(ie,1)) == idsDir{ibc}) ...
                sum(abs(Mesh.spig(ie,2)) == idsDir{ibc}) ...
                sum(abs(Mesh.spig(ie,3)) == idsDir{ibc})];
            if sum(onDir) > 0
                idOnDir = find(onDir==1);
                for i=1:length(idOnDir)
                    gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDir(i));
                    spigId = abs(Mesh.spig(ie,idOnDir(i)));
                    nodeId = Mesh.spig2(spigId,:);
                    intNode = Mesh.ele(ie, ...
                        Mesh.ele(ie,:)~=nodeId(1) & Mesh.ele(ie,:)~=nodeId(2));
                    l = diff(Mesh.node(nodeId,:));
                    n = cross([l 0],cross([l 0],...
                        [diff(Mesh.node([nodeId(1) intNode] ,:)) 0]));
                    n = n/norm(n);
                    frBC = zeros(Sys.pOrd+1,1);
                    for iq=1:length(wq1)
                        frBC = frBC + norm(l)*(NQuad1{iq}).'*wq1(iq);
                    end
                    Sys.f(gIs) = Sys.f(gIs) + frBC;
                    idx = s:s+(Sys.pOrd+1)^2-1;
                    for j=1:(Sys.pOrd+1)
                        for k = 1:(Sys.pOrd+1)
                            tmpIdx = idx((Sys.pOrd+1)*(j-1)+k);
                            IIDir(tmpIdx) = gIs(j);
                        end
                    end
                    s = s+(Sys.pOrd+1).^2;
                end
            end
        end
        Sys.Dir{ibc} = unique(IIDir(IIDir>0));
        DirIds = Sys.Dir{ibc};
        for jh=2:nHarms
            DirIds = [DirIds; Sys.Dir{ibc}+Sys.NDOFs*(jh-1)];
        end
        Sys.Dir{ibc} = DirIds;
    end
end


if FlagNeu
    NSNeu = length(idsNeu);
    IINeu = zeros(NSNeu*(Sys.pOrd+1)^2,1);
    s=1;
    for ie=1:Mesh.NELE
        onNeu = [sum(abs(Mesh.spig(ie,1)) == idsNeu) ...
            sum(abs(Mesh.spig(ie,2)) == idsNeu) ...
            sum(abs(Mesh.spig(ie,3)) == idsNeu)];
        if sum(onNeu) > 0
            idOnNeu = find(onNeu==1);
            for i=1:length(idOnNeu)
                spigId = abs(Mesh.spig(ie,idOnNeu(i)));
                nodeId = Mesh.spig2(spigId,:);
                intNode = Mesh.ele(ie, ...
                    Mesh.ele(ie,:)~=nodeId(1) & Mesh.ele(ie,:)~=nodeId(2));
                gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnNeu(i));
                l = diff(Mesh.node(nodeId,:));
                n = cross([l 0],cross([l 0],...
                    [diff(Mesh.node([nodeId(1) intNode] ,:)) 0]));
                n = n/norm(n);
                frBC = zeros(Sys.pOrd+1,1);
                for iq=1:length(wq1)
                    frBC = frBC + norm(l)*(NQuad1{iq}).'*wq1(iq);
                end
                idx = s:s+(Sys.pOrd+1)^2-1;
                for j=1:(Sys.pOrd+1)
                    for k = 1:(Sys.pOrd+1)
                        tmpIdx = idx((Sys.pOrd+1)*(j-1)+k);
                        IINeu(tmpIdx) = gIs(j);
                    end
                end
                s = s+(Sys.pOrd+1).^2;
                Sys.f(gIs) = Sys.f(gIs) + frBC;
            end
        end
    end
    Sys.Neu = unique(IINeu(IINeu>0));
end
%%
if FlagWP
    for ibc=1:length(idsWP)
        NSWP = length(idsWP{ibc});
        IIsWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        JJsWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        SSsWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        TTsWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        IIvWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        JJvWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        SSvWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        TTvWP = zeros(NSWP*(Sys.pOrd+1)^2,1);
        s=1;
        for ie=1:Mesh.NELE
            onDir = [sum(abs(Mesh.spig(ie,1)) == idsWP{ibc}) ...
                sum(abs(Mesh.spig(ie,2)) == idsWP{ibc}) ...
                sum(abs(Mesh.spig(ie,3)) == idsWP{ibc})];
            if sum(onDir) > 0
                idOnDir = find(onDir==1);
                Stt = zeros((Sys.pOrd+1),(Sys.pOrd+1));
                Ttt = zeros((Sys.pOrd+1),(Sys.pOrd+1));
                for i=1:length(idOnDir)
                    gIs = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDir(i));
                    spigId = abs(Mesh.spig(ie,idOnDir(i)));
                    nodeId = Mesh.spig2(spigId,:);
                    intNode = Mesh.ele(ie, ...
                        Mesh.ele(ie,:)~=nodeId(1) & Mesh.ele(ie,:)~=nodeId(2));
                    l = diff(Mesh.node(nodeId,:));
                    n = cross([l 0],cross([l 0],...
                        [diff(Mesh.node([nodeId(1) intNode] ,:)) 0]));
                    n = n/norm(n);
                    detJ = norm(l);
                    for iq=1:length(wq1)
                        Stt = Stt + norm(l)*((dNQuad1{iq}/detJ).'*(dNQuad1{iq}/detJ))*wq1(iq);
                        Ttt = Ttt + norm(l)*(NQuad1{iq}.'*NQuad1{iq})*wq1(iq);
                    end
                    idx = s:s+(Sys.pOrd+1)^2-1;
                    for j=1:(Sys.pOrd+1)
                        for k = 1:(Sys.pOrd+1)
                            tmpIdx = idx((Sys.pOrd+1)*(j-1)+k);
                            IIsWP(tmpIdx) = gIs(j);
                            JJsWP(tmpIdx) = gIs(k);
                            SSsWP(tmpIdx) = Stt(j,k);
                            TTsWP(tmpIdx) = Ttt(j,k);
                        end
                    end
                    s = s+(Sys.pOrd+1).^2;
                end
            end
        end
        SspWP = sparse(IIsWP,JJsWP,SSsWP,Sys.NDOFs,Sys.NDOFs);
        TspWP = sparse(IIsWP,JJsWP,TTsWP,Sys.NDOFs,Sys.NDOFs);
        Sys.WP{ibc} = unique(IIsWP(IIsWP>0));
        NDoFsWP = length(Sys.WP{ibc});
        for i=1:NDoFsWP
            for j=1:length(Sys.Dir)
                if(find(Sys.WP{ibc}(NDoFsWP-i+1) == Sys.Dir{j}))
                    Sys.WP{ibc}(NDoFsWP-i+1) = [];
                end
            end
        end
        SspWP = SspWP(Sys.WP{ibc},Sys.WP{ibc});
        TspWP = TspWP(Sys.WP{ibc},Sys.WP{ibc});
        [evect,eval] = eigs(SspWP,TspWP,Sys.WPnModes,'SM');
        [eval,idx] = sort(diag(eval),'ascend');
        evect = evect(:,idx);
%         idx = find(max(Mesh.refNode(Sys.WP{ibc},2)) == Mesh.refNode(Sys.WP{ibc},2));
%         pos = Mesh.refNode(Sys.WP{ibc},2);
%         [val, idx] =  sort(abs(pos - Mesh.refNode(Sys.WP{ibc}(idx(1)),2)));
%         figure; plot(Mesh.refNode(Sys.WP{ibc}(idx),2), evect(idx,:),'*-')
        % disp(evect.'*TspWP*evect)
        Sys.WPvec{ibc} = evect/sqrt(evect.'*TspWP*evect);
        Sys.WPfc{ibc} = sqrt(eval).'*Sys.c0/2/pi;
%         for jh=2:nHarms
%             Sys.WP{ibc} = [Sys.WP{ibc}; Sys.WP{ibc}+Sys.NDOFs*(jh-1)];
%             Sys.WPvec{ibc} = [Sys.WPvec{ibc}; evect];
%         end
    end
end
%%
Sys.bypass = true;
%fprintf('System matrices assembly: %2.4g s\n',toc);
end
