function D = GetCouplKerrFFT(nHarms, E, a, b, MtrlCx, FieldSx, posR, posL)
% Compute the harmonic-coupling matrix for Kerr nonlinearity
% using FFT of the time-domain permittivity (a + b*|E(t)|^2).
%
%   nHarms   : number of harmonics
%   E        : vector of |E| at each harmonic frequency
%   a, b     : linear permittivity and Kerr coefficient
%   MtrlCx   : cosine matrix [nHarms x N]
%   FieldSx  : sine matrix [nHarms x N]
%   posR, posL : right and left index positions in the FFT array
%
%   D : harmonic-coupling matrix [nHarms x nHarms]

N = size(MtrlCx, 2);

% FFT of the nonlinear permittivity in time
nlFFT = fft((a + b * (E.' * FieldSx).^2)) / N;

% Convolution with harmonic basis to extract coupling
couplFFT = fft(ones(nHarms, 1) * (nlFFT * MtrlCx) .* FieldSx ./ N, [], 2);
D = -imag(couplFFT(:, posL) - couplFFT(:, posR)).';

end
