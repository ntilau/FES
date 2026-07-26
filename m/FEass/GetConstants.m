function Sys = GetConstants(Sys)
% Set physical constants: speed of light, free-space impedance,
% permittivity, and permeability. Also define convenience handles
% for dB conversion and unwrapped angle.

Sys.c0  = 299792458;      % speed of light in vacuum [m/s]
Sys.z0  = 120 * pi;       % free-space impedance [Ohm]
Sys.eps0 = 1 / (Sys.z0 * Sys.c0);  % vacuum permittivity [F/m]
Sys.mu0 = Sys.z0 / Sys.c0;         % vacuum permeability [H/m]
Sys.db  = @(x) 20 * log10(abs(x));
Sys.arg = @(x) unwrap(angle(x)) * 180 / pi;

end
