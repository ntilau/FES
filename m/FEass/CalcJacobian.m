function [detJ, invJt] = CalcJacobian(xy)
% Compute Jacobian determinant and inverse-transpose of the
% mapping from reference to physical triangle.
%
%   xy : 3x2 matrix of vertex coordinates [x1 y1; x2 y2; x3 y3]

J    = [xy(2, 1) - xy(1, 1) xy(3, 1) - xy(1, 1); ...
        xy(2, 2) - xy(1, 2) xy(3, 2) - xy(1, 2)];
detJ = abs(det(J));
invJt = inv(J)';

end
