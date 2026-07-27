function [X, W] = CalcSimplexQuad(N, dim)
% Return Gauss-Legendre quadrature points and weights for an
% N-dimensional simplex (line: dim=1, triangle: dim=2).
%
%   N   : number of quadrature points per dimension (typically pOrd+1)
%   dim : spatial dimension (1 or 2)
%
% Based on Greg von Winckel's simplexquad.
%   http://math.unm.edu/~gregvw

if dim == 1
    [X, W] = rquad(N, 0);
    X = X(:);
    W = W(:);
elseif dim == 2
    % Build 2-D simplex quadrature from 1-D rules
    [q1, w1] = rquad(N, 1);
    [q2, w2] = rquad(N, 0);

    [Q1, Q2] = ndgrid(q1, q2);
    [W1, W2] = ndgrid(w1, w2);

    q  = [Q1(:), Q2(:)];
    w  = W1(:) .* W2(:);

    % Map from bi-unit square to triangle (Duffy transformation)
    X = [1 - q(:, 1), q(:, 1) .* (1 - q(:, 2)), q(:, 1) .* q(:, 2)];
    W = w .* q(:, 1);

    % Remove points below threshold
    keep = X(:, 1) >= 0 & X(:, 2) >= 0 & X(:, 3) >= 0;
    X = X(keep, :);
    W = W(keep);
else
    error('CalcSimplexQuad only supports dim=1 and dim=2');
end

end

function [x, w] = rquad(N, k)
% Compute Gauss-Radau quadrature nodes and weights.
k1  = k + 1;
k2  = k + 2;
n   = 1:N;
nnk = 2 * n + k;

A = [k / k2, repmat(k^2, 1, N) ./ (nnk .* (nnk + 2))];

n    = 2:N;
nnk  = nnk(n);
B1   = 4 * k1 / (k2 * k2 * (k + 3));
nk   = n + k;
nnk2 = nnk .* nnk;
B    = 4 * (n .* nk).^2 ./ (nnk2 .* nnk2 - nnk2);

ab = [A', [(2^k1) / k1; B1; B']];
s  = sqrt(ab(2:N, 2));

[V, X] = eig(diag(ab(1:N, 1), 0) + diag(s, -1) + diag(s, 1));
[X, I] = sort(diag(X));

x = (X + 1) / 2;
w = (1 / 2)^(k1) * ab(1, 2) * V(1, I)'.^2;

end
