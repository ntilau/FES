function WriteTwoPosts(a, c, r, L, n)
% Write a .poly geometry file for a rectangular waveguide
% with two cylindrical posts.
%   a : waveguide height
%   c : post centre-to-centre spacing
%   r : post radius
%   L : waveguide length
%   n : number of nodes per post perimeter

th = linspace(0, 2*pi*(n-1)/n, n);

polyID = fopen('mesh/TwoPosts.poly', 'w');
fprintf(polyID, '%d 2 1 1\n', 4+2*n);

% Waveguide corners
fprintf(polyID, '1 %g %g 0 %d\n', -L/2,  a/2, 11);
fprintf(polyID, '2 %g %g 0 %d\n',  L/2,  a/2, 12);
fprintf(polyID, '3 %g %g 0 %d\n',  L/2, -a/2, 12);
fprintf(polyID, '4 %g %g 0 %d\n', -L/2, -a/2, 11);

% Post perimeters (two posts)
for i = 1:n
    fprintf(polyID, '%d %g %g %d\n', 4+i, ...
        -(c/2) + r*cos(th(i)), r*sin(th(i)), 2);
end
for i = 1:n
    fprintf(polyID, '%d %g %g %d\n', 4+n+i, ...
        (c/2) + r*cos(th(i)), r*sin(th(i)), 2);
end

% Edge segments
fprintf(polyID, '%d 1\n', 4+2*n);

fprintf(polyID, '1 %d %d %d\n', 1, 2, 1);
fprintf(polyID, '2 %d %d %d\n', 3, 2, 12);
fprintf(polyID, '3 %d %d %d\n', 4, 3, 1);
fprintf(polyID, '4 %d %d %d\n', 1, 4, 11);

% Post perimeter edges
for i = 1:n-1
    fprintf(polyID, '%d %d %d %d\n', 4+i, 4+i+1, 4+i, 2);
end
fprintf(polyID, '%d %d %d %d\n', 4+n, 4+n, 4+1, 2);

for i = 1:n-1
    fprintf(polyID, '%d %d %d %d\n', 4+n+i, 4+n+i+1, 4+n+i, 2);
end
fprintf(polyID, '%d %d %d %d\n', 4+n+n, 4+n+n, 4+n+1, 2);

fprintf(polyID, '0\n');

% Region attributes
fprintf(polyID, '3\n');
fprintf(polyID, '1 %g %g %d\n', 0, 0, 1);
fprintf(polyID, '2 %g %g %d\n', -(c/2), 0, 2);
fprintf(polyID, '3 %g %g %d\n',  c/2, 0, 2);

fclose(polyID);
