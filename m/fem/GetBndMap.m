function [Mesh, Sys] = GetBndMap(Sys, Mesh)
% Build domain-decomposition boundary maps: identify boundary DoFs (BndDoF),
% region-specific DoFs (RegDoF), and the DDmap that renumbers interior
% region DoFs after boundary DoFs.

idsDD = find(Mesh.slab == Mesh.BC.DDschur);

nReg   = length(unique(Mesh.elab));
RegEle = cell(nReg, 1);
RegDoF = cell(nReg, 1);
RegDoFmap = cell(nReg, 1);
BndDoF = [];

%% Walk elements to collect boundary and region DoFs
for ie = 1:Mesh.NELE
    gIs = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
    tmp = Mesh.elab(ie);
    RegEle{tmp}  = unique([RegEle{tmp}; ie]);
    RegDoF{tmp}  = unique([RegDoF{tmp}; gIs.']);

    onDD = [sum(abs(Mesh.spig(ie, 1)) == idsDD) ...
            sum(abs(Mesh.spig(ie, 2)) == idsDD) ...
            sum(abs(Mesh.spig(ie, 3)) == idsDD)];
    if sum(onDD) > 0
        idOnDD = find(onDD == 1);
        for i = 1:length(idOnDD)
            spigId = Mesh.spig(ie, idOnDD(i));
            nodeId = Mesh.spig2(abs(spigId), :);
            gIs    = CalcGlobIndex(1, Sys.pOrd, Mesh, ie, idOnDD(i));
            BndDoF = [BndDoF gIs];
        end
    end
end
BndDoF = unique(BndDoF);

%% Build DD mapping: interior region DoFs shifted after boundary DoFs
Sys = CalcDoFsNumber(Sys, Mesh);

DDmap = zeros(Sys.NDOFs, 1);
roof  = length(BndDoF);
regOrder = 1:length(RegDoF);

for ir = regOrder
    RegDoF{ir}    = setdiff(RegDoF{ir}, BndDoF);
    RegDoFmap{ir} = roof + (1:length(RegDoF{ir})).';
    DDmap(RegDoF{ir}) = RegDoFmap{ir};
    roof = roof + length(RegDoF{ir});
end

DDmap(BndDoF) = 1:length(BndDoF);

Sys.DDmap     = DDmap;
Sys.BndDoF    = BndDoF.';
Sys.RegEle    = RegEle;
Sys.RegDoF    = RegDoF;
Sys.RegDoFmap = RegDoFmap;

end
