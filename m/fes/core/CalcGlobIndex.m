function [gIs, gIv] = CalcGlobIndex(dim, pOrd, Mesh, ie, is)
% Compute global DoF indices for element ie.
%
%   dim : 1 (edge) or 2 (element interior)
%   pOrd : polynomial order (1-4)
%   Mesh : mesh data structure
%   ie : element index (1-based)
%   is : local edge index (1-based, only used for dim=1)
%   gIs : scalar-field DoF indices (1-based)
%   gIv : vector-field (H(curl)) DoF indices (1-based, only for dim=2)
%
% For H(curl) elements, DoFs are organized as:
%   - Scalar DoFs: associated with nodes, edges, faces, interior
%   - Vector DoFs: associated with edges (tangential components)

NNODE = Mesh.NNODE;  % Number of nodes
NELE  = Mesh.NELE;   % Number of elements
NSPIG = Mesh.NSPIG;  % Number of segment edges (boundary edges)

switch dim
    case 1  % Edge DoFs (1D)
        % Get the global edge index for the local edge 'is' in element 'ie'
        gIv = abs(Mesh.spig(ie, is));
        % Get the two nodes that define this edge
        nIdx = Mesh.spig2(gIv, :);

        switch pOrd
            case 1
                % Linear edge: 2 nodes
                gIs = nIdx;
            case 2
                % Quadratic edge: 2 nodes + 1 edge midpoint
                gIs = [nIdx NNODE + abs(gIv)];
            case 3
                % Cubic edge: 2 nodes + 2 edge points
                gIs = [nIdx NNODE + abs(gIv) NNODE + NSPIG + abs(gIv)];
            case 4
                % Quartic edge: 2 nodes + 3 edge points
                gIs = [nIdx NNODE + abs(gIv) ...
                       NNODE + NSPIG + abs(gIv) ...
                       NNODE + 2 * NSPIG + NELE + abs(gIv)];
        end

    case 2  % Element DoFs (2D)
        % Get the three node indices of the element
        ele  = Mesh.ele(ie, :);
        % Get the three edge indices of the element (absolute values)
        spig = abs(Mesh.spig(ie, :));

        switch pOrd
            case 1
                % Linear triangle: 3 nodes (scalar), 3 edges (vector)
                gIs = ele;
                gIv = abs(Mesh.spig(ie, :));
            case 2
                % Quadratic triangle:
                %   Scalar: 3 vertices + 3 edge midpoints
                %   Vector: 3 edges + 3 edge midpoints + 2 interior + 1 element
                gIs = [ele NNODE + spig];
                gIv = [abs(Mesh.spig(ie, :)) ...
                       Mesh.NSPIG + abs(Mesh.spig(ie, :)) ...
                       2 * Mesh.NSPIG + ie ...
                       2 * Mesh.NSPIG + Mesh.NELE + ie];
            case 3
                % Cubic triangle:
                %   Scalar: 3 vertices + 2 edge points per edge + 1 interior
                %   Vector: 3 edges + 2 edge points per edge + 4 interior + 3 edge-interior + 2 element
                gIs = [ele NNODE + spig ...
                       NNODE + NSPIG + spig NNODE + 2 * NSPIG + ie];
                gIv = [abs(Mesh.spig(ie, :)) ...
                       Mesh.NSPIG + abs(Mesh.spig(ie, :)) ...
                       2 * Mesh.NSPIG + ie ...
                       2 * Mesh.NSPIG + Mesh.NELE + ie ...
                       2 * Mesh.NSPIG + 2 * Mesh.NELE + abs(Mesh.spig(ie, :)) ...
                       3 * Mesh.NSPIG + 2 * Mesh.NELE + ie ...
                       3 * Mesh.NSPIG + 3 * Mesh.NELE + ie ...
                       3 * Mesh.NSPIG + 4 * Mesh.NELE + ie ...
                       3 * Mesh.NSPIG + 5 * Mesh.NELE + ie];
            case 4
                % Quartic triangle:
                %   Scalar: 3 vertices + 3 edge points per edge + 3 interior
                %   Vector: 3 edges + 3 edge points per edge + 9 interior + 6 edge-interior + 3 element
                gIs = [ele NNODE + spig ...
                       NNODE + NSPIG + spig NNODE + 2 * NSPIG + ie ...
                       NNODE + 2 * NSPIG + NELE + spig ...
                       NNODE + 3 * NSPIG + NELE + ie ...
                       NNODE + 3 * NSPIG + 2 * NELE + ie];
                gIv = [abs(Mesh.spig(ie, :)) ...
                       Mesh.NSPIG + abs(Mesh.spig(ie, :)) ...
                       2 * Mesh.NSPIG + ie ...
                       2 * Mesh.NSPIG + Mesh.NELE + ie ...
                       2 * Mesh.NSPIG + 2 * Mesh.NELE + abs(Mesh.spig(ie, :)) ...
                       3 * Mesh.NSPIG + 2 * Mesh.NELE + ie ...
                       3 * Mesh.NSPIG + 3 * Mesh.NELE + ie ...
                       3 * Mesh.NSPIG + 4 * Mesh.NELE + ie ...
                       3 * Mesh.NSPIG + 5 * Mesh.NELE + ie];
        end
end

end
