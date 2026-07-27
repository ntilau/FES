function D = GetCouplFFT(Mtrl, Field)
% Compute harmonic-coupling matrix from a material function using FFT.
% Handles both sine/cosine basis and returns the coupling in a
% 2*nHarms x 2*nHarms form (interleaved sin/cos rows).

F(1:2:2 * (Mtrl.nHarms), :) = Field.Sin;
F(2:2:2 * (Mtrl.nHarms), :) = Field.Cos;

E(1:2:2 * (Mtrl.nHarms), :) = Field.E;
E(2:2:2 * (Mtrl.nHarms), :) = Field.E;

% FFT of the material response
nlFFT = fft(Mtrl.func(E.' * F)) / Mtrl.N;

% Convolution to extract coupling coefficients
couplFFT = fft(ones(2 * Mtrl.nHarms, 1) ...
    * (real(nlFFT) * Mtrl.Cos - imag(nlFFT) * Mtrl.Sin) ...
    .* (F / Mtrl.N), [], 2);

D = (couplFFT(:, Mtrl.posL) - couplFFT(:, Mtrl.posR)).';
D = -imag(D);

end
