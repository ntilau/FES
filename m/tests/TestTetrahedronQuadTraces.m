%%
clear all;% clc;
close all;
format short;
%geo  = [0 0 0; 1 0 0; 0 1 0; 0.5 0.5 0.5];
geo  = [0 0 0; 1 0 0; 0 1 0; 0 0 1];
% rdNs*geo

% [xq,wq]  = CalcSimplexQuad(2,3);
[xq,wq]  = CalcSimplexQuad(3,2);
% xq(:,2:3) = xq;
xq(:,3) = zeros(size(xq,1),1);
idx = [4 2 1];
idx2 = [4 2 1];
% shape = inline('[1-x-y-z x y z]');
% dShape = inline('[-1 1 0 0; -1 0 1 0; -1 0 0 1]','x','y','z');
[Shape, ShapeDerivX, ShapeDerivY, ShapeDerivZ] = CalcShapeFunctions(3, 1);
S = zeros(length(Shape(0,0,0)),length(Shape(0,0,0)));
T = S;
A = zeros(6,6);
K = A;
D = zeros(6,6);
G = zeros(6,6);
% n = 3\[1; 1; 1]*ones(1,6);
%     detJ = 1;
    n = [0 0 -1].'*ones(1,6);

figure(1); clf(1)
line([0 1 0 0 0 1 0 0 0],[0 0 1 0 0 0 1 0 0],[0 0 0 0 1 0 0 1 0]);
axis equal;

for iq=1:length(wq)
    %xq(iq,:)


    rNs = Shape(xq(iq,1),xq(iq,2),xq(iq,3));
    rdNs(1,:) = ShapeDerivX(xq(iq,1),xq(iq,2),xq(iq,3));
    rdNs(2,:) = ShapeDerivY(xq(iq,1),xq(iq,2),xq(iq,3));
    rdNs(3,:) = ShapeDerivZ(xq(iq,1),xq(iq,2),xq(iq,3));
    J = rdNs*geo;
    detJ = abs(det(J));
    rdNs = J\rdNs;

    rNv(:,1) = rNs(1)*rdNs(:,2)-rNs(2)*rdNs(:,1);
    rNv(:,2) = rNs(1)*rdNs(:,3)-rNs(3)*rdNs(:,1);
    rNv(:,3) = rNs(1)*rdNs(:,4)-rNs(4)*rdNs(:,1);
    rNv(:,4) = rNs(2)*rdNs(:,3)-rNs(3)*rdNs(:,2);
    rNv(:,5) = rNs(2)*rdNs(:,4)-rNs(4)*rdNs(:,2);
    rNv(:,6) = rNs(3)*rdNs(:,4)-rNs(4)*rdNs(:,3);
    rdNv(:,1) = 2*cross(rdNs(:,1),rdNs(:,2));
    rdNv(:,2) = 2*cross(rdNs(:,1),rdNs(:,3));
    rdNv(:,3) = 2*cross(rdNs(:,1),rdNs(:,4));
    rdNv(:,4) = 2*cross(rdNs(:,2),rdNs(:,3));
    rdNv(:,5) = 2*cross(rdNs(:,2),rdNs(:,4));
    rdNv(:,6) = 2*cross(rdNs(:,3),rdNs(:,4));

%
    shf = idx(2);
    figure(1);
    hold on;
    %quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rNv(1,1),rNv(2,1),rNv(3,1),1 );
    %quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rdNv(1,1),rdNv(2,1),rdNv(3,1),1 );
    quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rNv(1,shf),rNv(2,shf),0,.1,'color','r' );
    quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rdNv(2,shf),-rdNv(1,shf),0,.1,'color','k' );
    axis equal; xlabel('x'); ylabel('y'); zlabel('z');


%     cross(cross(n,rdNv),n)
    S = S + detJ*(rdNs.'*rdNs)*wq(iq);
    T = T + detJ*(rNs.'*rNs)*wq(iq);
    A = A + detJ*(dot(n,rdNv).'*dot(n,rdNv))*wq(iq);
    K = K + detJ*(cross(cross(n,rNv),n).'*cross(cross(n,rNv),n))*wq(iq);
    D = D + detJ*(cross(n,rdNv).'*cross(cross(n,rNv),n))*wq(iq);
    G = G + detJ*(cross(n,rdNv).'*cross(n,rdNv))*wq(iq);

end

% disp(A)
% disp(A(idx,idx))
disp(K(idx,idx))
disp(D(idx,idx))
% disp(G(idx,idx))
% D = D([1 2 4], [1,2 4])

% A = importdata('..\Bin\SS.txt', ' ', 3);
% disp(norm(S-A.data))
% A = importdata('..\Bin\TT.txt', ' ', 3);
% disp(norm(T-A.data))
