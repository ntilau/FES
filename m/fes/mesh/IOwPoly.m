function writePolyFile(a, b, D, W, flag)
% Write a .poly geometry file for a rectangular waveguide (optionally ridged).
%   a     : waveguide width
%   b     : waveguide height
%   D, W  : ridge dimensions (depends on flag)
%   flag  : 1 = single ridge, 2 = double ridge, omitted = plain rectangle

if nargin > 2
    switch flag
        case 1 % single ridge
            h = D;
            polyID = fopen('geom.poly', 'w');
            fprintf(polyID, '8 2 1 1\n');
            fprintf(polyID, '1 %g %g 0 %d\n', -a/2,  b/2,     1);
            fprintf(polyID, '2 %g %g 0 %d\n', -W/2,  b/2,     1);
            fprintf(polyID, '3 %g %g 0 %d\n', -W/2,  b/2-h,   1);
            fprintf(polyID, '4 %g %g 0 %d\n',  W/2,  b/2-h,   1);
            fprintf(polyID, '5 %g %g 0 %d\n',  W/2,  b/2,     1);
            fprintf(polyID, '6 %g %g 0 %d\n',  a/2,  b/2,     1);
            fprintf(polyID, '7 %g %g 0 %d\n',  a/2, -b/2,     1);
            fprintf(polyID, '8 %g %g 0 %d\n', -a/2, -b/2,     1);
            fprintf(polyID, '8 1\n');
            fprintf(polyID, '1 %d %d %d\n', 2, 1, 1);
            fprintf(polyID, '2 %d %d %d\n', 3, 2, 1);
            fprintf(polyID, '3 %d %d %d\n', 4, 3, 1);
            fprintf(polyID, '4 %d %d %d\n', 5, 4, 1);
            fprintf(polyID, '5 %d %d %d\n', 6, 5, 1);
            fprintf(polyID, '6 %d %d %d\n', 7, 6, 1);
            fprintf(polyID, '7 %d %d %d\n', 8, 7, 1);
            fprintf(polyID, '8 %d %d %d\n', 1, 8, 1);
            fprintf(polyID, '0\n');
            fprintf(polyID, '1\n');
            fprintf(polyID, '1 0 0 1\n');
            fclose(polyID);
        case 2 % double ridge
            S = D;
            polyID = fopen('geom.poly', 'w');
            fprintf(polyID, '12 2 1 1\n');
            fprintf(polyID, '1  %g %g 0 %d\n', -a/2,  b/2,   1);
            fprintf(polyID, '2  %g %g 0 %d\n', -W/2,  b/2,   1);
            fprintf(polyID, '3  %g %g 0 %d\n', -W/2,  S/2,   1);
            fprintf(polyID, '4  %g %g 0 %d\n',  W/2,  S/2,   1);
            fprintf(polyID, '5  %g %g 0 %d\n',  W/2,  b/2,   1);
            fprintf(polyID, '6  %g %g 0 %d\n',  a/2,  b/2,   1);
            fprintf(polyID, '7  %g %g 0 %d\n',  a/2, -b/2,   1);
            fprintf(polyID, '8  %g %g 0 %d\n',  W/2, -b/2,   1);
            fprintf(polyID, '9  %g %g 0 %d\n',  W/2, -S/2,   1);
            fprintf(polyID, '10 %g %g 0 %d\n', -W/2, -S/2,   1);
            fprintf(polyID, '11 %g %g 0 %d\n', -W/2, -b/2,   1);
            fprintf(polyID, '12 %g %g 0 %d\n', -a/2, -b/2,   1);
            fprintf(polyID, '12 1\n');
            fprintf(polyID, '1  %d %d %d\n', 2,  1,  1);
            fprintf(polyID, '2  %d %d %d\n', 3,  2,  1);
            fprintf(polyID, '3  %d %d %d\n', 4,  3,  1);
            fprintf(polyID, '4  %d %d %d\n', 5,  4,  1);
            fprintf(polyID, '5  %d %d %d\n', 6,  5,  1);
            fprintf(polyID, '6  %d %d %d\n', 7,  6,  1);
            fprintf(polyID, '7  %d %d %d\n', 8,  7,  1);
            fprintf(polyID, '8  %d %d %d\n', 9,  8,  1);
            fprintf(polyID, '9  %d %d %d\n', 10, 9,  1);
            fprintf(polyID, '10 %d %d %d\n', 11, 10, 1);
            fprintf(polyID, '11 %d %d %d\n', 12, 11, 1);
            fprintf(polyID, '12 %d %d %d\n', 1,  12, 1);
            fprintf(polyID, '0\n');
            fprintf(polyID, '1\n');
            fprintf(polyID, '1 0 0 1\n');
            fclose(polyID);
    end
else % plain rectangle
    polyID = fopen('geom.poly', 'w');
    fprintf(polyID, '4 2 1 1\n');
    fprintf(polyID, '1 %g %g 0 %d\n', -a/2,  b/2, 1);
    fprintf(polyID, '2 %g %g 0 %d\n',  a/2,  b/2, 1);
    fprintf(polyID, '3 %g %g 0 %d\n',  a/2, -b/2, 1);
    fprintf(polyID, '4 %g %g 0 %d\n', -a/2, -b/2, 1);
    fprintf(polyID, '4 1\n');
    fprintf(polyID, '1 %d %d %d\n', 2, 1, 1);
    fprintf(polyID, '2 %d %d %d\n', 3, 2, 1);
    fprintf(polyID, '3 %d %d %d\n', 4, 3, 1);
    fprintf(polyID, '4 %d %d %d\n', 1, 4, 1);
    fprintf(polyID, '0\n');
    fprintf(polyID, '1\n');
    fprintf(polyID, '1 0 0 1\n');
    fclose(polyID);
end
