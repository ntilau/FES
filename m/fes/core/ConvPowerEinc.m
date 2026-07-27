% Script to compute waveguide dimensions and incident power from
% electric field amplitude for standard rectangular waveguide (WR-90).
%
% This is a convenience script, not a function called at runtime.

% WR-90 waveguide dimensions [m]
a = 22.86e-03;
b = a / 2;           % b = a/2 for standard rectangular waveguide
% a = 19.05e-03;     % alternate: WR-75
% b = a / 2;

Einc = 5e3:5e3:15e3;   % incident E-field range [V/m]
c0   = 299792458;
Z0   = 376.730313461;   % free-space impedance [Ohm]
f    = 10e9;             % operating frequency [Hz]
fc   = c0 / (2 * a);     % TE10 cutoff [Hz]

% Incident power from E-field [W]
Pinc = Einc.^2 * (a * b * sqrt(1 - (fc / f).^2)) / (4 * Z0);

% E-field for unit power [V/m]
Einc = sqrt(1 ./ ((a * b * sqrt(1 - (fc / f).^2)) / (4 * Z0)));

format long e
Pinc
