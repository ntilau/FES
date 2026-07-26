function D = GetCouplSinFFT(nHarms, E, a, b, MtrlCx, FieldSx, posR, posL)
% Compute harmonic-coupling matrix for a sinusoidal Kerr nonlinearity
% using FFT, with sparse frequency-domain truncation.

N = size(MtrlCx, 2);

% FFT of the nonlinear permittivity
nlFFT = fft((a + b * (E.' * FieldSx).^2)) / N;

% Keep only significant Fourier coefficients
nl = real(nlFFT(1:size(MtrlCx, 1)));
idx = find(abs(nl) > 1e-13 * max(max(abs(nl))));
nl(2:end) = nl(2:end) * 2;

% Convolution with harmonic basis
Nx1  = ones(nHarms, 1) * (nl(idx) * MtrlCx(idx, :));
SxN  = FieldSx ./ N;
NxSx = Nx1 .* SxN;

couplFFT = fft(NxSx, [], 2);
D = 1i * (couplFFT(:, posL) - couplFFT(:, posR)).';

end
