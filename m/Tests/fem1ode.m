%% fem1ode - Solve a stiff FEM problem with time-dependent mass matrix
function fem1ode(N)

%% Setup
if nargin < 1
    N = 19;
end
h = pi / (N + 1);
y0 = sin(h * (1:N)');
tspan = [0; pi];

%% Build constant Jacobian and mass matrix
e = repmat(1 / h, N, 1);       % e = [(1/h) ... (1/h)];
d = repmat(-2 / h, N, 1);      % d = [(-2/h) ... (-2/h)];
J = spdiags([e d e], -1:1, N, N);

d = repmat(h / 6, N, 1);
M = spdiags([d 4 * d d], -1:1, N, N);

%% Solve ODE
options = odeset('Mass', @mass, 'MStateDep', 'none', ...
    'Jacobian', J);

[t, y] = ode23t(@f, tspan, y0, options);

%% Plot results
figure;
surf((1:N) / (N + 1), t, y);
set(gca, 'ZLim', [0 1]);
view(142.5, 30);
title(['Finite element problem with time-dependent mass ', ...
    'matrix, solved by ODE15S']);
xlabel('space ( x / \pi )');
ylabel('time');
zlabel('solution');

    %% Nested function: derivative
    function yp = f(t, y)
        yp = J * y;
    end

    %% Nested function: mass matrix
    function Mt = mass(t)
        Mt = exp(-t) * M;
    end

end
