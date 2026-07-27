%% PlotEfficiency - Plot solver efficiency and memory usage vs non-zero entries
clear all; clc; close all;

%% Data: model 1 (2 domains)
% DoFs = 700, NNZ = 8100,  Acc = 2.649, p = 1, 7732, 4644
% DoFs = 2908, NNZ = 52972,  Acc = 5.842, p = 2, 192864, 203868
% DoFs = 6640, NNZ = 176908, Acc = 10.11, p = 3, 645996, 682644
% DoFs = 11896, NNZ = 436572, Acc = 16.07, p = 4, 1415900, 1614068, tD = 216.275
mod1(1, :) = [700,   8100,   2.649,  1, 7732,   203868,  0];
mod1(2, :) = [2908,  52972,  5.842,  2, 192864, 203868,  0];
mod1(3, :) = [6640,  176908, 10.11,  3, 645996, 682644,  0];
mod1(4, :) = [11896, 436572, 16.07,  4, 1415900, 1614068, 216.275];

%% Data: model 2 (5 domains)
% DoFs = 1456, NNZ = 12980, Acc = 4.393, p = 1, 32940, 37480
% DoFs = 6032, NNZ = 86176, Acc = 9.085, p = 2, 471576, 477836
% DoFs = 13744, NNZ = 289000, Acc = 16.82, p = 3, 1250820, 1364400
% DoFs = 24592, NNZ = 714476, Acc = 28.2, p = 4, 2893748, 3210944, 268
mod2(1, :) = [1456,  12980,  4.393,  1, 32940,  37480,   0];
mod2(2, :) = [6032,  86176,  9.085,  2, 471576, 477836,  0];
mod2(3, :) = [13744, 289000, 16.82,  3, 1250820, 1364400, 0];
mod2(4, :) = [24592, 714476, 28.2,   4, 2893748, 3210944, 268];

%% Plot acceleration vs non-zero entries
figure;
plot(mod1(:, 2), mod1(:, 3), '.-', mod2(:, 2), mod2(:, 3), '-o')
xlabel('Non-zero entries');
ylabel('Acceleration');
legend('2 domains', '5 domains', 'Location', 'SouthEast');

%% Plot memory allocation vs non-zero entries
figure;
plot(mod1(:, 2), mod1(:, 5), '.-', ...
    mod1(:, 2), mod1(:, 6), '-o', ...
    mod2(:, 2), mod2(:, 5), '.-.', ...
    mod2(:, 2), mod2(:, 6), '-.o');
xlabel('Non-zero entries');
ylabel('Allocated memory (kB)');
legend('Full - 2 domains', 'DD - 2 domains', ...
    'Full - 5 domains', 'DD - 5 domains', ...
    'Location', 'SouthEast');
