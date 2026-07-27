function WriteCirculator(a, s, nt1, d)
% Write a .poly geometry file for a Y-junction ferrite circulator.
%   a    : ferrite post radius
%   s    : scaling factor relative to a 10 GHz design
%   nt1  : number of nodes on the post perimeter
%   d    : mesh density

thb = linspace(0, 2*pi*(nt1-1)/nt1, nt1);

polyID = fopen('mesh/circulator.poly', 'w');
fprintf(polyID, '%d 2 1 1\n', 9+nt1);

% Nodes
fprintf(polyID, '1 %g %g %d %d 2\n', -39.9991*s, 11.43*s, 0, 11);
fprintf(polyID, '2 %g %g %d %d 2\n', -6.5991*s, 11.43*s, 0, 1);
fprintf(polyID, '3 %g %g %d %d 2\n', 10.1009*s, 40.3552*s, 0, 12);
fprintf(polyID, '4 %g %g %d %d 2\n', 29.8982*s, 28.9252*s, 0, 12);
fprintf(polyID, '5 %g %g %d %d 2\n', 13.0424*s, 0, 0, 1);
fprintf(polyID, '6 %g %g %d %d 2\n', 29.8982*s, -28.9252*s, 0, 13);
fprintf(polyID, '7 %g %g %d %d 2\n', 10.1009*s, -40.3552*s, 0, 13);
fprintf(polyID, '8 %g %g %d %d 2\n', -6.5991*s, -11.43*s, 0, 1);
fprintf(polyID, '9 %g %g %d %d 2\n', -39.9991*s, -11.43*s, 0, 11);
for i = 1:nt1
    fprintf(polyID, '%d %g %g %d\n', 9+i, ...
        a*cos(thb(i)), a*sin(thb(i)), 0);
end

% Edge segments
nedges = 9 + nt1;
fprintf(polyID, '%d 1\n', nedges);
for i = 1:2
    fprintf(polyID, '%d %d %d %d\n', i, i+1, i, 1);
end
fprintf(polyID, '%d %d %d %d\n', 3, 4, 3, 12);
for i = 4:5
    fprintf(polyID, '%d %d %d %d\n', i, i+1, i, 1);
end
fprintf(polyID, '%d %d %d %d\n', 6, 7, 6, 13);
for i = 7:8
    fprintf(polyID, '%d %d %d %d\n', i, i+1, i, 1);
end
fprintf(polyID, '%d %d %d %d\n', 9, 1, 9, 11);
for i = 10:nedges-1
    fprintf(polyID, '%d %d %d %d\n', i, i+1, i, 2);
end
fprintf(polyID, '%d %d %d %d\n', nedges, 10, nedges, 2);

% Hole definition (none)
fprintf(polyID, '0\n');

% Region attributes with mesh density
fprintf(polyID, '2\n');
fprintf(polyID, '1 %g %g %d %g\n', -a-0.1, 0, 1, 22.86*s/d);
fprintf(polyID, '2 %g %g %d %g\n', 0, 0, 2, a/d);

fclose(polyID);
