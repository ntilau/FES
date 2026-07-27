% solvers
clc; close all; clear
% Accuracy
% Change default axes fonts.
set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14)

% Change default text fonts.
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 14)

set(0,'defaultlinelinewidth', 1);

set(gcf, 'PaperUnits', 'centimeters');
%set(gcf, 'PaperType', 'A4');
set(gcf, 'PaperSize', [8 10]);
td = [0.842 2.823 13.119 64.475 213.5 1672.05];
memd = [55.0586 177.142 797.28 3365.86 9638.4 48789.9];
dofd = [18214 58710 163612 528494 1286582 4194110];
ti = [26.925 104.3 623.052 3265.34 11564.4 82458.2];
memi = [73.4414 212.5 571.312 1785.11 4045.3 14320.3];
dofi = [22803 69392 186186 581737 1303938 4433964];

figure;

[ax,h1,h2] = plotyy(dofi,ti,dofi,memi,@loglog);
set(get(ax(1),'Ylabel'),'String','Time [s]')
set(get(ax(2),'Ylabel'),'String','Memory [MB]')
xlabel('Number of unknowns')
set(h1, 'LineStyle','--','Marker','<')
set(h2, 'LineStyle','-','Marker','x','MarkerSize',8)
% axis tight
% legend('DD-GMRES time', 'DD-GMRES mem.')
% hold on;
hold on
plot(dofi,0.000007*dofi.^1.5,'k-.', dofi,0.003*dofi,'k-.')

% return
figure

[ax,h1,h2] = plotyy(dofd,td,dofd,memd,@loglog);
set(get(ax(1),'Ylabel'),'String','Time [s]')
set(get(ax(2),'Ylabel'),'String','Memory [MB]')
xlabel('Number of unknowns')
set(h1, 'LineStyle','--','Marker','o')
set(h2, 'LineStyle','-','Marker','+')
hold on
plot(dofd,0.0000007*dofd.^1.4,'k-.', dofd,0.000000000005*dofd.^2.5,'k-.')

% legend( 'Direct time', 'Direct mem.')
