function Mesh = CalcDoFsPositions(Sys, Mesh)
% Compute reference coordinates (refNode) and element connectivity
% (refEle) for each degree of freedom at given polynomial order.

Mesh.refNode = zeros(Sys.NDOFs, 3);
Mesh.refEle  = zeros(Mesh.NELE * Sys.pOrd^2, 3);

for ie = 1:Mesh.NELE
    gIdx = CalcGlobIndex(2, Sys.pOrd, Mesh, ie);
    nIdx = Mesh.ele(ie, :);
    sIdx = abs(Mesh.spig(ie, :));

    % Corner nodes (always exist)
    Mesh.refNode(gIdx(1:3), :) = [Mesh.node(nIdx, :) zeros(3, 1)];

    switch Sys.pOrd
        case 1
            Mesh.refEle(ie, :) = nIdx;

        case 2
            nIdx = Mesh.spig2(sIdx, :);
            % Edge midpoint nodes
            Mesh.refNode(gIdx(4:6), :) = [1/2 * Mesh.node(nIdx(:, 1), :) ...
                + 1/2 * Mesh.node(nIdx(:, 2), :) zeros(3, 1)];
            % Sub-triangle connectivity
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
