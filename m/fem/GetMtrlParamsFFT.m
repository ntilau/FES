function Mtrl = GetMtrlParamsFFT(Sys)
% Compute FFT parameters for harmonic-balance material coupling:
% number of FFT bins, time samples, frequency line, and index positions
% for extracting the desired harmonics.

if ~isfield(Sys, 'OverSampling')
    Sys.OverSampling = 4;
end

Mtrl.nHarms = length(Sys.HBharms);
f1 = Sys.freq * Sys.HBharms(1);
f2 = Sys.freq * Sys.HBharms(2);
df = min([f1 f2 abs(round(f1 - f2))]);

Mtrl.bf = gcd(df, f1);

% Number of FFT bins (odd, to avoid half-bin shift)
N = ceil(max(Sys.HBharms) * Sys.OverSampling * f1 / Mtrl.bf);
Mtrl.N = N + mod(N + 1, 2);

% Time and frequency vectors
Mtrl.t   = linspace(0, 1 / Mtrl.bf * (Mtrl.N - 1) / Mtrl.N, Mtrl.N);
Mtrl.fnl = Mtrl.bf * [0 1:floor(Mtrl.N / 2) -floor(Mtrl.N / 2):-1];

% Index positions for extracting positive/negative frequency components
Mtrl.posL = uint32(Sys.freq * Sys.HBharms / Mtrl.bf) + 1;
Mtrl.posR = Mtrl.N - uint32(Sys.freq * Sys.HBharms / Mtrl.bf) + 1;

end
