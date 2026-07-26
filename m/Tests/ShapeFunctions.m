clear all;
syms x0 x1 x2 x3
hierarchical = false;
dim = 3;
x0 = 1-x1-x2-x3;
if(hierarchical)
    shape(20) = x2*x0*x1;
    shape(19) = x0*x1*x3;
    shape(18) = x0*x2*x3;
    shape(17) = x1*x2*x3;
    shape(16) = x2*x3*(x2-x3);
    shape(15) = x1*x3*(x1-x3);
    shape(14) = x1*x2*(x1-x2);
    shape(13) = (-x3+x0)*x3*x0;
    shape(12) = x0*x2*(x0-x2);
    shape(11) = x0*x1*(x0-x1);
    shape(10) = 4*x2*x3;
    shape(9) = 4*x1*x3;
    shape(8) = 4*x1*x2;
    shape(7) = 4*x0*x3;
    shape(6) = 4*x0*x2;
    shape(5) = 4*x1*x0;
    shape(4) = x3;
    shape(3) = x2;
    shape(2) = x1;
    shape(1) = x0;
else
    switch dim
        case 2
            shape(10) = 4*x2*x3;
            shape(9) = 4*x1*x3;
            shape(8) = 4*x1*x2;
            shape(7) = 4*x0*x3;
            shape(6) = 4*x0*x2;
            shape(5) = 4*x1*x0;
            shape(4) = x3*(2*x3-1);
            shape(3) = x2*(2*x2-1);
            shape(2) = x1*(2*x1-1);
            shape(1) = x0*(2*x0-1);
        case 3
            shape(20) = 27*x2*x0*x1;
            shape(19) = 27*x0*x1*x3;
            shape(18) = 27*x0*x2*x3;
            shape(17) = 27*x1*x2*x3;
            shape(16) = 9/2*x3*x2*(3*x3-1);
            shape(15) = 9/2*x2*x3*(3*x2-1);
            shape(14) = 9/2*x3*x1*(3*x3-1);
            shape(13) = 9/2*x1*x3*(3*x1-1);
            shape(12) = 9/2*x2*x1*(3*x2-1);
            shape(11) = 9/2*x1*x2*(3*x1-1);
            shape(10) = 9/2*x3*x0*(3*x3-1);
            shape(9) = 9/2*x0*x3*(3*x0-1);
            shape(8) = 9/2*x2*x0*(3*x2-1);
            shape(7) = 9/2*x0*x2*(3*x0-1);
            shape(6) = 9/2*x1*x0*(3*x1-1);
            shape(5) = 9/2*x0*x1*(3*x0-1);
            shape(4) = 1/2*x3*(3*x3-1)*(3*x3-2);
            shape(3) = 1/2*x2*(3*x2-1)*(3*x2-2);
            shape(2) = 1/2*x1*(3*x1-1)*(3*x1-2);
            shape(1) = 1/2*x0*(3*x0-1)*(3*x0-2);
    end
end

points = [0 0 0
    1 0 0
    0 1 0
    0 0 1
    1/3 0 0
    0 1/3 0
    0 0 1/3
    2/3 1/3 0
    2/3 0 1/3
    0 2/3 1/3
    2/3 0 0
    0 2/3 0
    0 0 2/3
    1/3 2/3 0
    1/3 0 2/3
    0 1/3 2/3
    1/3 1/3 1/3
    0 1/3 1/3
    1/3 0 1/3
    1/3 1/3 0];

idx = 1:20;
%idx = [0,1,2,3,4,6,8,10,12,14,5,7,9,11,13,15,16,17,18,19] +1;
%points = points(idx([1 2 3 4 8 9 10 7 6 5 14 15 16 13 12 11 17 18 19 20]),:);
%points = points(idx([0,1,2,3,4,6,8,10,12,14,5,7,9,11,13,15,16,17,18,19]+1),:);
%points = points(1+[0,1,2,3,4,6,8,10,12,14,5,7,9,11,13,15,16,17,18,19],:);

% format short
% for i=1:20
%     f = inline(shape(i),'x1','x2','x3');
%     g(i,:) = (f(points(:,1),points(:,2),points(:,3)).');
% end
% [row,col] = find(g);
%%
%row = idx([1 2 3 4 8 9 10 7 6 5 14 15 16 13 12 11 17 18 19 20]);
% fprintf('{')
% for i=1:20
%     fprintf('%d',idx(row(i)-1);
%     if(i<20)
%         fprintf(',');
%     end
% end
% fprintf('}\n');

% return
Jacobian = simplify(jacobian(shape.').');

disp(shape.')
disp('grad')
for i=1:length(shape)
disp(Jacobian(:,i))
end
