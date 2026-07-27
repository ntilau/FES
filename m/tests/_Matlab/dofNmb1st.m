%% dof numbering 1st order
clear
% save dofNmb1st
load dofNmb1st;
dof = lnDebugES.';
dof = dof(:)+1;
pnt = -ones(size(dof));

k = 1;
for i = 1:length(dof)
    idx = find(dof == dof(i) & pnt ~= -1);
    if idx
        pnt(i) = pnt(idx(1));
    else
        pnt(i) = k;
        k = k + 1;
    end
end
pnt = pnt - 1;
