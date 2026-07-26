%% trace operator on 2D
clear all;% clc;
format short;
% geo  = [0 0 0; 1 0 0; 0 1 0; 0 0 1];

geo  = [0 0; 1 0; 0 1];

[xq2, wq2] = CalcSimplexQuad(3,2);
% shape = inline('[1-x-y-z x y z]');
% dShape = inline('[-1 1 0 0; -1 0 1 0; -1 0 0 1]','x','y','z');
[Shape, ShapeDerivX, ShapeDerivY] = CalcShapeFunctions(2, 1);
numv = 3;

Sv = zeros(numv);
Tv = Sv;
D = Sv;
G = Sv;
detJ = 1;

% figure(1); clf(1)
% line([0 1 0 0 0 1 0 0 0],[0 0 1 0 0 0 1 0 0],[0 0 0 0 1 0 0 1 0]);
% axis equal;

for iq=1:length(wq2)
    x = Shape(xq2(iq,1),xq2(iq,2));
    dx(1,:) = ShapeDerivX(xq2(iq,1),xq2(iq,2));
    dx(2,:) = ShapeDerivY(xq2(iq,1),xq2(iq,2));
    switch 1
        case 1
            Nv2(:,1) = x(2)*dx(:,3)-x(3)*dx(:,2);
            Nv2(:,2) = x(1)*dx(:,3)-x(3)*dx(:,1);
            Nv2(:,3) = x(1)*dx(:,2)-x(2)*dx(:,1);
            dNv2(:,1) = 2;
            dNv2(:,2) = -2;
            dNv2(:,3) = 2;
        case 2
            Nv2(:,1) = x(2)*dx(:,3)-x(3)*dx(:,2);
            Nv2(:,2) = x(1)*dx(:,3)-x(3)*dx(:,1);
            Nv2(:,3) = x(1)*dx(:,2)-x(2)*dx(:,1);
            Nv2(:,4) = 4*(x(2)*dx(:,3)+x(3)*dx(:,2));
            Nv2(:,5) = 4*(x(1)*dx(:,3)+x(3)*dx(:,1));
            Nv2(:,6) = 4*(x(1)*dx(:,2)+x(2)*dx(:,1));
            Nv2(:,7) = (x(1)*x(2)*dx(:,3)-x(1)*x(3)*dx(:,2));
            Nv2(:,8) = (x(1)*x(2)*dx(:,3)-x(2)*x(3)*dx(:,1));
            dNv2(:,1) = 2;
            dNv2(:,2) = -2;
            dNv2(:,3) = 2;
            dNv2(:,4) = 0;
            dNv2(:,5) = 0;
            dNv2(:,6) = 0;
            dNv2(:,7) = 2*x(1)-x(2)-x(3);
            dNv2(:,8) = x(1)+x(3)-2*x(2);
        case 3
            Nv2(:,1) = x(2)*dx(:,3)-x(3)*dx(:,2);
            Nv2(:,2) = x(1)*dx(:,3)-x(3)*dx(:,1);
            Nv2(:,3) = x(1)*dx(:,2)-x(2)*dx(:,1);
            Nv2(:,4) = 4*(x(2)*dx(:,3)+x(3)*dx(:,2));
            Nv2(:,5) = 4*(x(1)*dx(:,3)+x(3)*dx(:,1));
            Nv2(:,6) = 4*(x(1)*dx(:,2)+x(2)*dx(:,1));
            Nv2(:,7) = (x(1)*x(2)*dx(:,3)-x(1)*x(3)*dx(:,2));
            Nv2(:,8) = (x(1)*x(2)*dx(:,3)-x(2)*x(3)*dx(:,1));
            Nv2(:,9) = x(2)*(x(2)-2*x(3))*dx(:,3) + x(3)*(-x(3)+2*x(2)) * dx(:,2);
            Nv2(:,10) = x(1)*(x(1)-2*x(3))*dx(:,3) +  x(3)*(-x(3)+2*x(1)) * dx(:,1);
            Nv2(:,11) = x(1)*(x(1)-2*x(2))*dx(:,2) + x(2)*(-x(2)+2*x(1)) * dx(:,1);
            Nv2(:,12) = x(2)*x(1)*dx(:,3) + x(1)*x(3)*dx(:,2) + x(2)*x(3)*dx(:,1);
            Nv2(:,13) = -x(2)*x(1)*(x(2)-2*x(3)) * dx(:,3) -x(1)*x(3)*(-x(3)+2*x(2))*dx(:,2) + 3*x(2)*x(3)*(x(2)-x(3))*dx(:,1);
            Nv2(:,14) = -x(1)*x(2)*(x(1)-2*x(3))*dx(:,3) + 3*x(1)*x(3)*(x(1)-x(3))*dx(:,2) - x(2)*x(3)*(-x(3)+2*x(1))*dx(:,1);
            Nv2(:,15) = 3*x(1)*x(2)*(x(1)-x(2))*dx(:,3) - x(1)*x(3)*(x(1)-2*x(2))*dx(:,2) -x(2)*x(3)*(-x(2)+2*x(1))*dx(:,1);
            dNv2(:,1) = 2;
            dNv2(:,2) = -2;
            dNv2(:,3) = 2;
            dNv2(:,4) = 0;
            dNv2(:,5) = 0;
            dNv2(:,6) = 0;
            dNv2(:,7) = 2*x(1)-x(2)-x(3);
            dNv2(:,8) = x(1)+x(3)-2*x(2);
            dNv2(:,9) = 0;
            dNv2(:,10) = 0;
            dNv2(:,11) = 0;
            dNv2(:,12) = 0;
            dNv2(:,13) = -16*x(2)*x(3)+4*x(3)*x(3)+4*x(2)*x(2);
            dNv2(:,14) = 16*x(1)*x(3)-4*x(3)*x(3)-4*x(1)*x(1);
            dNv2(:,15) = -16*x(2)*x(1)+4*x(2)*x(2)+4*x(1)*x(1);
        otherwise
            error( 'order must be between 1..3' );
    end

%         shf = 1;
%     figure(1);
%     hold on;
%     %quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rNv(1,1),rNv(2,1),rNv(3,1),1 );
%     %quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rdNv(1,1),rdNv(2,1),rdNv(3,1),1 );
%     quiver3(xq2(iq,1),xq2(iq,2),0, Nv2(1,shf),Nv2(2,shf),0,1,'color','r' );
%     quiver3(xq2(iq,1),xq2(iq,2),0, Nv2(2,shf),-Nv2(1,shf),0,1,'color','k' );
%     axis equal;

    a = [0 0; 0 -2; -2 0];
    dNv2 = dNv2 * (dx(1,2)*dx(2,3)-dx(2,2)*dx(1,3));
    Sv = Sv + detJ*(dNv2.'*dNv2)*wq2(iq);
    Tv = Tv + detJ*(Nv2.'*Nv2)*wq2(iq);
    D = D + detJ*(a*Nv2)*wq2(iq);
%     D = D + detJ*( (Nv2.'*[0 -1; 1 0]) * Nv2)*wq2(iq);
%     G = G + detJ*( (Nv2.'*[0 -1; 1 0]) * (Nv2.'*[0 -1; 1 0]).')*wq2(iq);
%     D = D + .5*detJ*( (dNv2.'*[1 -1]) * Nv2)*wq2(iq);
%     G = G + .5*detJ*( (dNv2.'*[1 -1]) * (dNv2.'*[1 -1]).')*wq2(iq);
end
% disp(Sv)
disp(D)
% disp(Tv)
% disp(D)
% disp(G)
% disp([         0         0         0
%     0.3333    0.6667    0.3333
%    -0.3333    0.3333    0.6667])
