%%
clear all;% clc;
format short;
geo  = [0 0 0; 1 0 0; 0 1 0; 0 0 1];

[xq,wq]  = CalcSimplexQuad(2,3);
%[xq,wq]  = CalcSimplexQuad(3,3);
% xq = [1 0 0; 0 1 0; 0 1 0; 1 1 0; 1 0 1;0 1 1]/2;
% wq = [1 1 1 1 1 1]/6;

% xq(:,2:3) = xq;
% xq(:,3) = zeros(size(xq,1),1);
idx = [4 2 1];
% idx = [6 5 4];
%idx = [6 3 2];
% shape = inline('[1-x-y-z x y z]');
% dShape = inline('[-1 1 0 0; -1 0 1 0; -1 0 0 1]','x','y','z');
[Shape, ShapeDerivX, ShapeDerivY, ShapeDerivZ] = CalcShapeFunctions(3, 1);
S = zeros(length(Shape(0,0,0)),length(Shape(0,0,0)));
T = S;
A = zeros(6,6);
K = A;
D = zeros(6,6);
n = [0; 0; -1]*ones(1,6);
    detJ = 1;

figure(1);clf(1)
line([0 1 0 0 0 1 0 0 0],[0 0 1 0 0 0 1 0 0],[0 0 0 0 1 0 0 1 0]);
axis equal;

for iq=1:length(wq)
    %xq(iq,:)
    geo  = [0 0 0; 1 0 0; 0 1 0; 0 0 1];

    rNs = Shape(xq(iq,1),xq(iq,2),xq(iq,3));
    rdNs(1,:) = ShapeDerivX(xq(iq,1),xq(iq,2),xq(iq,3));
    rdNs(2,:) = ShapeDerivY(xq(iq,1),xq(iq,2),xq(iq,3));
    rdNs(3,:) = ShapeDerivZ(xq(iq,1),xq(iq,2),xq(iq,3));
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


    shf = 1;%idx(3);
    figure(1);
    hold on;
    quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rNv(1,shf),rNv(2,shf),rNv(3,shf),0.1,'color','r' );
    quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rdNv(1,shf),rdNv(2,shf),rdNv(3,shf),0.1,'color','k' );
%     quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rNv(1,shf),rNv(2,shf),0,1,'color','r' );
%     quiver3(xq(iq,1),xq(iq,2),xq(iq,3), rNv(2,shf),-rNv(1,shf),0,1,'color','k' );
    axis equal; xlabel('x'); ylabel('y'); zlabel('z');

    %J = rdNs*geo;
%     detJ = 1;%abs(det(J));
%     rdNs = J\rdNs;

    S = S + detJ*(rdNs.'*rdNs)*wq(iq);
    T = T + detJ*(rNs.'*rNs)*wq(iq);
    A = A + detJ*(rdNv.'*rdNv)*wq(iq);
    K = K + detJ*(rNv.'*rNv)*wq(iq);
    D = D + detJ*(rdNv.'*rNv)*wq(iq);

end

% disp(A)
disp(D(idx,idx))
% D = D([1 2 4], [1,2 4])

% A = importdata('..\Bin\SS.txt', ' ', 3);
% disp(norm(S-A.data))
% A = importdata('..\Bin\TT.txt', ' ', 3);
% disp(norm(T-A.data))
