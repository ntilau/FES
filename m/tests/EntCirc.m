%% EntCirc - Create a circular geometry entity
%  Ent = EntCirc(center, radius)
function Ent = EntCirc(center, radius)
    Ent.type = 'circle';
    toll = 1e-2;
    N = floor(2 * pi * radius / toll);
    theta = linspace(0, 2 * pi * (N - 1) / N, N);
    Ent.points(:, 1) = center(1) + radius * cos(theta);
    Ent.points(:, 2) = center(2) + radius * sin(theta);
    Ent.edges(:, 1) = 1:N;
    Ent.edges(:, 2) = [2:N, 1];
end
