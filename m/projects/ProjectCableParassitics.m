%% PrjCableParassitics - Test geometry entities with concentric circles
Config();

% 2D geometry descriptor
geom{1} = EntCirc([0, 0], 0.1);
geom{2} = EntCirc([0, 0], 0.2);

figure;
for i = 1:length(geom)
    plot(geom{i}.points(:, 1), geom{i}.points(:, 2), 'k.-')
    hold on;
end
axis equal;
axis tight;
