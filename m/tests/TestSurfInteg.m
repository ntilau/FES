% Surface integration for tetrahedron field


geoRef = [0 0 0; 1 0 0; 0 1 0; 0 0 1];

geo = [
 -0.0186  0.0375   0.0391
  0.0035  0.0161   0.0391
  0.0094  0.0375   0.0391
  0.0099  0.0281   0.0227
  ];

v0 = geo(2,:);
v1 = geo(3,:);
v2 = geo(4,:);
intNode = geo(1,:);

vIn = ((v0+v1+v2)/3.0) - intNode;
vIn = vIn / norm(vIn);

v1 = v1 - v0;
v2 = v2 - v0;
u = v1 / norm(v1);
n = cross(v1,v2);
n = n * dot(n,vIn);
n = n / norm(n);
v = cross(n,u);

figure; plot3(geo(:,1),geo(:,2),geo(:,3),'o')
hold on;
quiver3(0,0,0,v0(1),v0(2),v0(3),1)
quiver3(v0(1),v0(2),v0(3),v1(1),v1(2),v1(3),1)
quiver3(v0(1),v0(2),v0(3),v2(1),v2(2),v2(3),1)
% quiver3(v0(1),v0(2),v0(3),u(1),u(2),u(3),1)
% quiver3(v0(1),v0(2),v0(3),v(1),v(2),v(3),1)
% quiver3(v0(1),v0(2),v0(3),n(1),n(2),n(3),1)

axis equal
locNode = [6.666666666666666e-001,    1.666666666666667e-001, 1.666666666666667e-001];
pnt = v0 + locNode(1)*v1 + locNode(2)*v2;
plot3(pnt(:,1),pnt(:,2),pnt(:,3),'or')
locNode = [1.666666666666667e-001,    6.666666666666666e-001, 1.666666666666667e-001];
pnt = v0 + locNode(1)*v1 + locNode(2)*v2;
plot3(pnt(:,1),pnt(:,2),pnt(:,3),'og')
locNode = [1.666666666666667e-001,    1.666666666666667e-001, 6.666666666666666e-001];
pnt = v0 + locNode(1)*v1 + locNode(2)*v2;
plot3(pnt(:,1),pnt(:,2),pnt(:,3),'ob')
