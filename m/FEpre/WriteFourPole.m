function writeFourPole(a, l1, l2, l3, l4, w1, w2, w3, w4, w5, t, ...
    r1, r2, r3, r4, nt1, nt2)
% Write a .poly geometry file for a four-pole waveguide filter.
%   a         : waveguide height
%   l1..l4    : iris (post) lengths
%   w1..w5    : cavity widths
%   t         : iris thickness
%   r1..r4    : post rounding radii
%   nt1, nt2  : number of perimeter nodes for rounded features

thb = linspace(0, 2*pi*(nt1-1)/nt1, nt1);
ths = linspace(0, 2*pi*(nt2-1)/nt2, nt2);

polyID = fopen('fourPole.poly', 'w');
fprintf(polyID, '%d 2 1 1\n', 44+2*nt2+2*nt1);
d = -(2.5*t + l1 + l2 + l1 + l2);

% --- Top profile (left to right) ---
fprintf(polyID, '1  %g %g 0 %d\n', d,               a/2,  11);
fprintf(polyID, '2  %g %g 0 %d\n', d+l1+l2,         a/2,   0);
d = d + l1 + l2;
fprintf(polyID, '3  %g %g 0 %d\n', d,               w1/2,  0);
fprintf(polyID, '4  %g %g 0 %d\n', d+t,             w1/2,  0);
fprintf(polyID, '5  %g %g 0 %d\n', d+t,             a/2,   0);
fprintf(polyID, '6  %g %g 0 %d\n', d+t+l1,          a/2,   0);
d = d + t + l1;
fprintf(polyID, '7  %g %g 0 %d\n', d,               w2/2,  0);
fprintf(polyID, '8  %g %g 0 %d\n', d+t,             w2/2,  0);
fprintf(polyID, '9  %g %g 0 %d\n', d+t,             a/2,   0);
fprintf(polyID, '10 %g %g 0 %d\n', d+t+l2,          a/2,   0);
d = d + t + l2;
fprintf(polyID, '11 %g %g 0 %d\n', d,               w3/2,  0);
fprintf(polyID, '12 %g %g 0 %d\n', d+t,             w3/2,  0);
fprintf(polyID, '13 %g %g 0 %d\n', d+t,             a/2,   0);
fprintf(polyID, '14 %g %g 0 %d\n', d+t+l3,          a/2,   0);
d = d + t + l3;
fprintf(polyID, '15 %g %g 0 %d\n', d,               w4/2,  0);
fprintf(polyID, '16 %g %g 0 %d\n', d+t,             w4/2,  0);
fprintf(polyID, '17 %g %g 0 %d\n', d+t,             a/2,   0);
fprintf(polyID, '18 %g %g 0 %d\n', d+t+l4,          a/2,   0);
d = d + t + l4;
fprintf(polyID, '19 %g %g 0 %d\n', d,               w5/2,  0);
fprintf(polyID, '20 %g %g 0 %d\n', d+t,             w5/2,  0);
fprintf(polyID, '21 %g %g 0 %d\n', d+t,             a/2,   0);
fprintf(polyID, '22 %g %g 0 %d\n', d+t+l4+l3,       a/2,  12);

% --- Far right, bottom profile (right to left) ---
fprintf(polyID, '23 %g %g 0 %d\n', d+t+l4+l3,      -a/2, 12);
fprintf(polyID, '24 %g %g 0 %d\n', d+t,            -a/2,  0);
fprintf(polyID, '25 %g %g 0 %d\n', d+t,            -w5/2, 0);
fprintf(polyID, '26 %g %g 0 %d\n', d,              -w5/2, 0);
d = d - t - l4;
fprintf(polyID, '27 %g %g 0 %d\n', d+t+l4,         -a/2,  0);
fprintf(polyID, '28 %g %g 0 %d\n', d+t,            -a/2,  0);
fprintf(polyID, '29 %g %g 0 %d\n', d+t,            -w4/2, 0);
fprintf(polyID, '30 %g %g 0 %d\n', d,              -w4/2, 0);
d = d - t - l3;
fprintf(polyID, '31 %g %g 0 %d\n', d+t+l3,         -a/2,  0);
fprintf(polyID, '32 %g %g 0 %d\n', d+t,            -a/2,  0);
fprintf(polyID, '33 %g %g 0 %d\n', d+t,            -w3/2, 0);
fprintf(polyID, '34 %g %g 0 %d\n', d,              -w3/2, 0);
d = d - t - l2;
fprintf(polyID, '35 %g %g 0 %d\n', d+t+l2,         -a/2,  0);
fprintf(polyID, '36 %g %g 0 %d\n', d+t,            -a/2,  0);
fprintf(polyID, '37 %g %g 0 %d\n', d+t,            -w2/2, 0);
fprintf(polyID, '38 %g %g 0 %d\n', d,              -w2/2, 0);
d = d - t - l1;
fprintf(polyID, '39 %g %g 0 %d\n', d+t+l1,         -a/2,  0);
fprintf(polyID, '40 %g %g 0 %d\n', d+t,            -a/2,  0);
fprintf(polyID, '41 %g %g 0 %d\n', d+t,            -w1/2, 0);
fprintf(polyID, '42 %g %g 0 %d\n', d,              -w1/2, 0);
d = d - l1 - l2;
fprintf(polyID, '43 %g %g 0 %d\n', d+l1+l2,        -a/2,  0);
fprintf(polyID, '44 %g %g 0 %d\n', d,              -a/2, 11);

% --- Cylindrical iris posts (rounded corners) ---
for i = 1:nt1
    fprintf(polyID, '%d %g %g %d\n', 44+i, ...
        -(1.5*t + 0.5*l1 + l2) + r1*cos(thb(i)), r1*sin(thb(i)), 2);
end
for i = 1:nt2
    fprintf(polyID, '%d %g %g %d\n', 44+nt1+i, ...
        -(0.5*t + 0.5*l2) + r2*cos(ths(i)), r2*sin(ths(i)), 2);
end
for i = 1:nt2
    fprintf(polyID, '%d %g %g %d\n', 44+nt1+nt2+i, ...
        (0.5*t + 0.5*l2) + r3*cos(ths(i)), r3*sin(ths(i)), 2);
end
for i = 1:nt1
    fprintf(polyID, '%d %g %g %d\n', 44+nt1+2*nt2+i, ...
        (1.5*t + 0.5*l1 + l2) + r4*cos(thb(i)), r4*sin(thb(i)), 2);
end

% --- Edge segments ---
fprintf(polyID, '%d 1\n', 44+2*nt2+2*nt1);
for i = 1:21
    fprintf(polyID, '%d %d %d %d\n', i, i+1, i, 1);
end
fprintf(polyID, '22 %d %d %d\n', 22, 23, 12);
for i = 23:43
    fprintf(polyID, '%d %d %d %d\n', i, i+1, i, 1);
end
fprintf(polyID, '44 %d %d %d\n', 44, 1, 11);

% Post perimeter edges (circles)
for i = 1:nt1-1
    fprintf(polyID, '%d %d %d %d\n', 44+i, 44+i+1, 44+i, 2);
end
fprintf(polyID, '%d %d %d %d\n', 44+nt1, 44+nt1, 44+1, 2);

for i = 1:nt2-1
    fprintf(polyID, '%d %d %d %d\n', ...
        44+nt1+i, 44+nt1+i+1, 44+nt1+i, 2);
end
fprintf(polyID, '%d %d %d %d\n', ...
    44+nt1+nt2, 44+nt1+nt2, 44+nt1+1, 2);

for i = 1:nt2-1
    fprintf(polyID, '%d %d %d %d\n', ...
        44+nt1+nt2+i, 44+nt1+nt2+i+1, 44+nt1+nt2+i, 2);
end
fprintf(polyID, '%d %d %d %d\n', ...
    44+nt1+2*nt2, 44+nt1+2*nt2, 44+nt1+nt2+1, 2);

for i = 1:nt1-1
    fprintf(polyID, '%d %d %d %d\n', ...
        44+nt1+2*nt2+i, 44+nt1+2*nt2+i+1, 44+nt1+2*nt2+i, 2);
end
fprintf(polyID, '%d %d %d %d\n', ...
    44+nt1+nt2+nt2+nt1, 44+nt1+2*nt2+nt1, 44+nt1+2*nt2+1, 2);

fprintf(polyID, '0\n');

% Region attributes
fprintf(polyID, '5\n');
fprintf(polyID, '1 %g %g %d 0.5\n', 0, 0, 1);
fprintf(polyID, '2 %g %g %d 0.1\n', -(1.5*t + 0.5*l1 + l2), 0, 2);
fprintf(polyID, '3 %g %g %d 0.1\n', -(0.5*t + 0.5*l2),       0, 2);
fprintf(polyID, '4 %g %g %d 0.1\n',  (0.5*t + 0.5*l2),       0, 2);
fprintf(polyID, '5 %g %g %d 0.1\n',  (1.5*t + 0.5*l1 + l2),  0, 2);

fclose(polyID);
