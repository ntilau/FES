function [Shape, ShapeDerivX, ShapeDerivY, ShapeDerivZ] = CalcShapeFunctions(dim, p)
% computation of scalar basis functions of order p
% needs symbolic math
%clc; clear;
% for dim = 1:3
%     for p = 1:2
        syms x y z f;
        Shape = ' ';
        ShapeDerivX = ' ';
        ShapeDerivY = ' ';
        ShapeDerivZ = ' ';
        switch dim
            case 1
                l0 = 1-x;
                l1 = x;
                switch p
                    case 1
                        f(1) = l0;
                        f(2) = l1;
                    case 2
                        f(1) = l0*(2*l0-1);
                        f(2) = l1*(2*l1-1);
                        f(3) = 4*l0*l1;
                    case 3
                        f(1) = 1/2*(3*l0-1)*(3*l0-2)*l0;
                        f(2) = 1/2*(3*l1-1)*(3*l1-2)*l1;
                        f(3) = 9/2*(3*l0-1)*l0*l1;
                        f(4) = 9/2*(3*l1-1)*l1*l0;
                    case 4
                        f(1) = 1/3*(4*l0-1)*(2*l0-1)*(4*l0-3)*l0;
                        f(2) = 1/3*(4*l1-1)*(2*l1-1)*(4*l1-3)*l1;
                        f(3) = 16/3*(4*l0-1)*(2*l0-1)*l0*l1;
                        f(4) = 4*(4*l0-1)*(4*l1-1)*l0*l1;
                        f(5) = 16/3*(4*l1-1)*(2*l1-1)*l0*l1;
                end
                disp([dim p]);
                disp(simplify(f));
                disp(simplify(diff(f.').'));
                % figure; ezplot(f(3),[0 1])
                Shape = char(simplify(f));
                Shape = inline(Shape(9:end-2),'x');
                ShapeDerivX = char(simplify(diff(f.').'));
                ShapeDerivX = inline(ShapeDerivX(9:end-2),'x');
            case 2
                l0 = 1-x-y;
                l1 = x;
                l2 = y;
                switch p
                    case 1
                        f(1) = l0;
                        f(2) = l1;
                        f(3) = l2;
                    case 2
                        f(1) = l0*(2*l0-1);
                        f(2) = l1*(2*l1-1);
                        f(3) = l2*(2*l2-1);
                        f(4) = 4*l1*l2;
                        f(5) = 4*l2*l0;
                        f(6) = 4*l0*l1;
                    case 3
                        f(1) = 1/2*(3*l0-1)*(3*l0-2)*l0;
                        f(2) = 1/2*(3*l1-1)*(3*l1-2)*l1;
                        f(3) = 1/2*(3*l2-1)*(3*l2-2)*l2;
                        f(4) = 9/2*(3*l1-1)*l1*l2;
                        f(5) = 9/2*(3*l2-1)*l1*l2;
                        f(6) = 9/2*(3*l2-1)*l0*l2;
                        f(7) = 9/2*(3*l0-1)*l0*l2;
                        f(8) = 9/2*(3*l0-1)*l0*l1;
                        f(9) = 9/2*(3*l1-1)*l1*l0;
                        f(10) = 27*l0*l1*l2;
                        f = f([1 2 3 4 7 8 5 6 9 10]);
                    case 4
                        f(1) = 1/3*(4*l0-1)*(2*l0-1)*(4*l0-3)*l0;
                        f(2) = 1/3*(4*l1-1)*(2*l1-1)*(4*l1-3)*l1;
                        f(3) = 1/3*(4*l2-1)*(2*l2-1)*(4*l2-3)*l2;
                        f(4) = 16/3*(4*l1-1)*(2*l1-1)*l1*l2;
                        f(5) = 4*(4*l1-1)*(4*l2-1)*l1*l2;
                        f(6) = 16/3*(4*l2-1)*(2*l2-1)*l1*l2;
                        f(7) = 16/3*(4*l2-1)*(2*l2-1)*l0*l2;
                        f(8) = 4*(4*l2-1)*(4*l0-1)*l0*l2;
                        f(9) = 16/3*(4*l0-1)*(2*l0-1)*l0*l2;
                        f(10) = 16/3*(4*l0-1)*(2*l0-1)*l0*l1;
                        f(11) = 4*(4*l0-1)*(4*l1-1)*l0*l1;
                        f(12) = 16/3*(4*l1-1)*(2*l1-1)*l0*l1;
                        f(13) = 32*(4*l0-1)*l0*l1*l2;
                        f(14) = 32*(4*l1-1)*l0*l1*l2;
                        f(15) = 32*(4*l2-1)*l0*l1*l2;
                        f = f([1 2 3 4 9 10 5 8 11 13 6 7 12 14 15]);
                end
                disp([dim p]);
                disp(simplify(f));
                disp(simplify(jacobian(f.')).');
                % figure; ezsurf(f(5),[0 1 0 1])
                Shape = char(simplify(f));
                Shape = inline(Shape(9:end-2),'x','y');
                Jacobian = simplify(jacobian(f.').');
                ShapeDerivX = char(Jacobian(1,:));
                ShapeDerivX = inline(ShapeDerivX(9:end-2),'x','y');
                ShapeDerivY = char(Jacobian(2,:));
                ShapeDerivY = inline(ShapeDerivY(9:end-2),'x','y');
            case 3
                l0 = 1-x-y-z;
                l1 = x;
                l2 = y;
                l3 = z;
                switch p
                    case 1
                        f(1) = l0;
                        f(2) = l1;
                        f(3) = l2;
                        f(4) = l3;
                    case 2
                        f(1) = l0*(2*l0-1);
                        f(2) = l1*(2*l1-1);
                        f(3) = l2*(2*l2-1);
                        f(4) = l3*(2*l3-1);
                        f(5) = 4*l0*l1;
                        f(6) = 4*l0*l2;
                        f(7) = 4*l0*l3;
                        f(8) = 4*l1*l2;
                        f(9) = 4*l1*l3;
                        f(10) = 4*l2*l3;
                    case 3
                        f(20) = 27*l2*l0*l1;
                        f(19) = 27*l0*l1*l3;
                        f(18) = 27*l0*l2*l3;
                        f(17) = 27*l1*l2*l3;
                        f(16) = 9/2*l3*l2*(3*l3-1);
                        f(15) = 9/2*l2*l3*(3*l2-1);
                        f(14) = 9/2*l3*l1*(3*l3-1);
                        f(13) = 9/2*l1*l3*(3*l1-1);
                        f(12) = 9/2*l2*l1*(3*l2-1);
                        f(11) = 9/2*l1*l2*(3*l1-1);
                        f(10) = 9/2*l3*l0*(3*l3-1);
                        f(9) = 9/2*l0*l3*(3*l0-1);
                        f(8) = 9/2*l2*l0*(3*l2-1);
                        f(7) = 9/2*l0*l2*(3*l0-1);
                        f(6) = 9/2*l1*l0*(3*l1-1);
                        f(5) = 9/2*l0*l1*(3*l0-1);
                        f(4) = 1/2*l3*(3*l3-1)*(3*l3-2);
                        f(3) = 1/2*l2*(3*l2-1)*(3*l2-2);
                        f(2) = 1/2*l1*(3*l1-1)*(3*l1-2);
                        f(1) = 1/2*l0*(3*l0-1)*(3*l0-2);
                        %f = f([1 2 3 4 7 8 5 6 9 10]);
                    case 4
                        f(1) = 1/3*(4*l0-1)*(2*l0-1)*(4*l0-3)*l0;
                        f(2) = 1/3*(4*l1-1)*(2*l1-1)*(4*l1-3)*l1;
                        f(3) = 1/3*(4*l2-1)*(2*l2-1)*(4*l2-3)*l2;
                        f(4) = 16/3*(4*l1-1)*(2*l1-1)*l1*l2;
                        f(5) = 4*(4*l1-1)*(4*l2-1)*l1*l2;
                        f(6) = 16/3*(4*l2-1)*(2*l2-1)*l1*l2;
                        f(7) = 16/3*(4*l2-1)*(2*l2-1)*l0*l2;
                        f(8) = 4*(4*l2-1)*(4*l0-1)*l0*l2;
                        f(9) = 16/3*(4*l0-1)*(2*l0-1)*l0*l2;
                        f(10) = 16/3*(4*l0-1)*(2*l0-1)*l0*l1;
                        f(11) = 4*(4*l0-1)*(4*l1-1)*l0*l1;
                        f(12) = 16/3*(4*l1-1)*(2*l1-1)*l0*l1;
                        f(13) = 32*(4*l0-1)*l0*l1*l2;
                        f(14) = 32*(4*l1-1)*l0*l1*l2;
                        f(15) = 32*(4*l2-1)*l0*l1*l2;
                        f = f([1 2 3 4 9 10 5 8 11 13 6 7 12 14 15]);
                end
                disp([dim p]);
                disp(simplify(f));
                disp(simplify(jacobian(f.')).');
                % figure; ezsurf(f(5),[0 1 0 1])
                Shape = char(simplify(f));
                Shape = inline(Shape(9:end-2),'x','y','z');
                Jacobian = simplify(jacobian(f.').');
                ShapeDerivX = char(Jacobian(1,:));
                ShapeDerivX = inline(ShapeDerivX(9:end-2),'x','y','z');
                ShapeDerivY = char(Jacobian(2,:));
                ShapeDerivY = inline(ShapeDerivY(9:end-2),'x','y','z');
                ShapeDerivZ = char(Jacobian(3,:));
                ShapeDerivZ = inline(ShapeDerivZ(9:end-2),'x','y','z');

        end
%     end
% end

