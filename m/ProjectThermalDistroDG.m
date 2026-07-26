%% ProjectThermalDistroDG - Transient heat equation with DG in time
clear; close all;
Config();

%% Setup
Sys.pOrd = 1;
Sys.hOrd = 1;

Mesh = IOrPoly('ModelHeatEquation', 'q34a0.01A', Sys.hOrd, 1);

Mesh.BC.Dir = 2;
Mesh.BC.Neu = 3;   % flux condition
Sys.ht = 750;
Sys.K = 100;
Sys.RhoCp = 1000000;   % J/m^3/K
Sys.Tedge{1} = 300;
Sys.Tedge{2} = 200;
Sys.Text = 100;
Sys.q0 = 100000;

[Sys, Mesh] = AssembleDG(Sys, Mesh);

%% Initial conditions
u1 = ones(Sys.NDOF, 1) * Sys.Text;
if isfield(Sys, 'Dir')
    for ibc = 1:length(Sys.Dir)
        u1(Sys.Dir{ibc}) = Sys.Tedge{ibc};
    end
end

%% Time-stepping loop
t = 0:1:1000;
Dt = abs(t(2) - t(1));

for i = 1:length(t)
    A = Sys.RhoCp / Dt * Sys.T;
    b = Sys.q0 * Sys.f - Sys.K * Sys.S * u1 ...
        + Sys.RhoCp / Dt * Sys.T * u1;

    %% Apply Dirichlet conditions
    if isfield(Sys, 'Dir')
        for ibc = 1:length(Sys.Dir)
            b = b - A(:, Sys.Dir{ibc}) ...
                * ones(length(Sys.Dir{ibc}), 1) * Sys.Tedge{ibc};
        end
        for ibc = 1:length(Sys.Dir)
            b(Sys.Dir{ibc}) = Sys.Tedge{ibc};
            A(Sys.Dir{ibc}, :) = 0;
            A(:, Sys.Dir{ibc}) = 0;
            A(Sys.Dir{ibc}, Sys.Dir{ibc}) = eye(length(Sys.Dir{ibc}));
        end
    end

    u = A \ b;

    %% Plot
    fig = figure(1);
    pdeplot(Mesh.refNode.', [], Mesh.refEle.', ...
        'xydata', u, 'zdata', u, 'mesh', 'on', ...
        'colormap', 'hot', 'xygrid', 'on');
    title(['t = ', num2str(t(i)), '/', num2str(max(t))])
    view(0, 45)

    u1 = u;
end
