function Mesh = BuildRegularSquare(nptsx, nptsy)
% Build a regular triangular mesh on the unit square.
tic;
[ele, node] = regular_triangular_mesh(nptsx, nptsy);
NNODE = size(node, 1);
NELE = size(ele, 1);
nlab = zeros(NNODE, 1);
nlab(node(:,1) == 0 | node(:,1) == 1 | node(:,2) == 0 | node(:,2) == 1) = 1;
elab = ones(NELE, 1);
tspig = [];
slab = [];
for ie = 1:NELE
    ele(ie,:) = sort(ele(ie,:));
    nodes = ele(ie,:);
    for i = 1:3
        snodes = sort([nodes(i) nodes(mod(i, 3) + 1)]);
        if isempty(tspig)
            tspig = [tspig snodes.'];
            if (nlab(snodes(1)) == 1) && (nlab(snodes(2)) == 1) ...
                    && sum(diff(node(snodes,:)) == 0)
                slab = [slab; 1];
            else
                slab = [slab; 0];
            end
        else
            if find(tspig(1,:) == snodes(1) & tspig(2,:) == snodes(2))
            elseif find(tspig(1,:) == snodes(2) & tspig(2,:) == snodes(1))
            else
                tspig = [tspig snodes.'];
                if (nlab(snodes(1)) == 1) && (nlab(snodes(2)) == 1) ...
                        && sum(diff(node(snodes,:)) == 0)
                    slab = [slab; 1];
                else
                    slab = [slab; 0];
                end
            end
        end
    end
end
spig2 = tspig.';
NSPIG = length(spig2);
spig = zeros(NELE, 3);
for ie = 1:NELE
    nodes = ele(ie,:);
    for i = 1:3
        snodes = [nodes(mod(i, 3) + 1) nodes(mod(i + 1, 3) + 1)];
        if isempty(find(tspig(1,:) == snodes(1) & tspig(2,:) == snodes(2), 1))
            spig(ie, i) = -find(tspig(1,:) == snodes(2) ...
                & tspig(2,:) == snodes(1));
        else
            spig(ie, i) = find(tspig(1,:) == snodes(1) ...
                & tspig(2,:) == snodes(2));
        end
    end
end
Mesh.node = node;
Mesh.ele = ele;
Mesh.spig = spig;
Mesh.spig2 = spig2;
Mesh.nlab = nlab;
Mesh.elab = elab;
Mesh.slab = slab;
Mesh.NNODE = NNODE;
Mesh.NELE = NELE;
Mesh.NSPIG = NSPIG;
fprintf('Regular mesh generation: %2.4g s\n', toc);

end


function [T, V] = regular_triangular_mesh(nx, ny)
% Create a regular triangular tiling of the unit square.
[x, y] = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny));
V = [x(:) y(:)];
idx = reshape(1:prod([ny, nx]), ny, nx);
v1 = idx(1:end-1, 1:end-1); v1 = v1(:);
v2 = idx(1:end-1, 2:end);   v2 = v2(:);
v3 = idx(2:end,   1:end-1); v3 = v3(:);
v4 = idx(2:end,   2:end);   v4 = v4(:);
T = [v1 v2 v4; v1 v3 v4];
end
