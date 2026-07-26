function PlotPoly(FileName, figHandle, scale)
% Plot the geometry defined in a .poly file (Triangle format).
polyID = fopen(['FEpre/', FileName, '.poly'], 'r');
rows = fgetl(polyID);
rown = sscanf(rows, '%d');
Geom.NNODE = rown(1);
Geom.dim = rown(2);
Geom.node = zeros(Geom.NNODE, Geom.dim);
Geom.nlab = zeros(Geom.NNODE, 1);
for i = 1:Geom.NNODE
    rows = fgetl(polyID);
    rown = sscanf(rows, '%g');
    Geom.node(rown(1), :) = rown(2:1+Geom.dim);
    Geom.nlab(rown(1)) = rown(end);
end
if nargin > 2
    Geom.node = Geom.node * scale;
end
rows = fgetl(polyID);
rown = sscanf(rows, '%g');
Geom.NSPIG = rown(1);
Geom.spig = zeros(Geom.NSPIG, 2);
Geom.slab = zeros(Geom.NSPIG, 1);
for i = 1:Geom.NSPIG
    rows = fgetl(polyID);
    rown = sscanf(rows, '%g');
    Geom.spig(rown(1), :) = rown(2:3);
    Geom.slab(rown(1)) = rown(end);
end
fclose(polyID);
figure(figHandle);
hold on;
for i = 1:Geom.NSPIG
    line(Geom.node(Geom.spig(i,:), 1), ...
         Geom.node(Geom.spig(i,:), 2), 'Color', 'k');
end
axis equal; axis tight;
