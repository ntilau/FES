function [Mtrl, Field] = GetECouplFFT(Mtrl, Field)
% Compute the E-field coupling matrix (Kerr nonlinearity) via FFT.
% Returns Mtrl.D as a 2*nHarms x 2*nHarms matrix with interleaved
% sine (odd rows) and cosine (even rows) components.

F(1:2:2 * (Mtrl.nHarms), :) = Field.Sin;
F(2:2:2 * (Mtrl.nHarms), :) = Field.Cos;

% FFT of the nonlinear permittivity function
nlFFT = fft(Mtrl.func(abs(Field.E).' * F)) / Mtrl.N;

% Convolution to extract sine/cosine coupling
couplFFT = fft(ones(2 * Mtrl.nHarms, 1) ...
    * ((nlFFT) * (Mtrl.Cos - 1i * Mtrl.Sin)) .* (F / Mtrl.N), [], 2);

Mtrl.Dsin = 1i * (couplFFT(:, Mtrl.posL) - couplFFT(:, Mtrl.posR)).';
Mtrl.Dcos = (couplFFT(:, Mtrl.posL) + couplFFT(:, Mtrl.posR)).';

Mtrl.D(1:2:2 * (Mtrl.nHarms), :) = Mtrl.Dsin;
Mtrl.D(2:2:2 * (Mtrl.nHarms), :) = Mtrl.Dcos;

end
