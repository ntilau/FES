function [D, N] = GetCouplSinAnalyt(nHarm, e, a, b)
% Compute harmonic-coupling matrix for a sinusoidal Kerr nonlinearity
% analytically. Only uses odd-indexed entries of e (cosine amplitudes).
%
%   nHarm : number of harmonics
%   e     : harmonic amplitudes (including zeros at even indices)
%   a, b  : linear and nonlinear coefficients
%
% NOTE: This code was symbolically derived. The numeric constants use
% MATLAB's floating-point representation (e.g., 0.2e1 = 2.0).

N = eye(nHarm);

switch nHarm
    case 1
        D = a + b * e ^ 2;

    case 2
        e = [e(1) 0 e(2) 0];
        nl1 = b * e(3)^2 / 2 + b * e(4)^2 / 2 + a ...
            + b * e(2)^2 / 2 + b * e(1)^2 / 2;
        nl3 = -b * (-2 * e(4) * e(2) - 2 * e(1) * e(3) + e(1)^2 - e(2)^2) / 2;
        nl5 = -b * (-e(4) * e(2) + e(1) * e(3));
        nl7 = -b * (-e(4)^2 + e(3)^2) / 2;

        D(1, 1) = nl1 - nl3 / 2;
        D(2, 1) = nl3 / 2 - nl5 / 2;
        D(1, 2) = D(2, 1);
        D(2, 2) = nl1 - nl7 / 2;

    case 3
        e = [e(1) 0 e(2) 0 e(3) 0];
        nl1 = b * e(3)^2 / 2 + b * e(5)^2 / 2 + b * e(2)^2 / 2 ...
            + b * e(4)^2 / 2 + b * e(6)^2 / 2 + b * e(1)^2 / 2 + a;
        nl3 = -b * (-2 * e(3) * e(1) - 2 * e(4) * e(2) - 2 * e(3) * e(5) ...
            - e(2)^2 + e(1)^2 - 2 * e(6) * e(4)) / 2;
        nl5 = -b * (-e(6) * e(2) + e(3) * e(1) - e(4) * e(2) - e(1) * e(5));
        nl7 = -b * (-e(4)^2 - 2 * e(6) * e(2) + 2 * e(1) * e(5) + e(3)^2) / 2;
        nl9 = -b * (-e(6) * e(4) + e(3) * e(5));

        D(1, 1) = nl1 - nl3 / 2;
        D(2, 1) = nl3 / 2 - nl5 / 2;
        D(3, 1) = -nl7 / 2 + nl5 / 2;
        D(1, 2) = D(2, 1);
        D(2, 2) = nl1 - nl7 / 2;
        D(3, 2) = nl3 / 2 - nl9 / 2;
        D(1, 3) = D(3, 1);
        D(2, 3) = D(3, 2);
        D(3, 3) = nl1;

    case 4
        e = [e(1) 0 e(2) 0 e(3) 0 e(4) 0];
        nl1  = b * e(1)^2 / 2 + b * e(5)^2 / 2 + b * e(3)^2 / 2 ...
             + b * e(7)^2 / 2 + a;
        nl3  = -b * e(1)^2 / 2 + b * e(1) * e(3) + b * e(3) * e(5) + b * e(5) * e(7);
        nl5  = -b * e(1) * e(3) + b * e(1) * e(5) + b * e(3) * e(7);
        nl7  = -b * e(1) * e(5) + b * e(1) * e(7) - b * e(3)^2 / 2;
        nl9  = -b * e(1) * e(7) - b * e(3) * e(5);
        nl11 = -b * e(3) * e(7) - b * e(5)^2 / 2;
        nl13 = -b * e(5) * e(7);
        nl15 = -b * e(7)^2 / 2;

        D(1, 1) = nl1 - nl3 / 2;
        D(2, 1) = -nl5 / 2 + nl3 / 2;
        D(3, 1) = nl5 / 2 - nl7 / 2;
        D(4, 1) = -nl9 / 2 + nl7 / 2;
        D(1, 2) = D(2, 1);
        D(2, 2) = nl1 - nl7 / 2;
        D(3, 2) = nl3 / 2 - nl9 / 2;
        D(4, 2) = nl5 / 2 - nl11 / 2;
        D(1, 3) = D(3, 1);
        D(2, 3) = D(3, 2);
        D(3, 3) = -nl11 / 2 + nl1;
        D(4, 3) = nl3 / 2;
        D(1, 4) = D(4, 1);
        D(2, 4) = D(4, 2);
        D(3, 4) = D(4, 3);
        D(4, 4) = nl1;

    case 5
        e = [e(1) 0 e(2) 0 e(3) 0 e(4) 0 e(5) 0];
        nl1  = a + b * e(3)^2 / 2 + b * e(5)^2 / 2 + b * e(7)^2 / 2 ...
             + b * e(9)^2 / 2 + b * e(1)^2 / 2;
        nl3  = -b * e(1)^2 / 2 + b * e(1) * e(3) + b * e(7) * e(9) ...
             + b * e(3) * e(5) + b * e(5) * e(7);
        nl5  = -b * e(1) * e(3) + b * e(3) * e(7) + b * e(5) * e(9) + b * e(1) * e(5);
        nl7  = b * e(1) * e(7) - b * e(3)^2 / 2 - b * e(1) * e(5) + b * e(3) * e(9);
        nl9  = b * e(1) * e(9) - b * e(1) * e(7) - b * e(3) * e(5);
        nl11 = -b * e(5)^2 / 2 - b * e(1) * e(9) - b * e(3) * e(7);
        nl13 = -b * e(5) * e(7) - b * e(3) * e(9);
        nl15 = -b * e(5) * e(9) - b * e(7)^2 / 2;
        nl17 = -b * e(7) * e(9);
        nl19 = -b * e(9)^2 / 2;

        D(1, 1) = nl1 - nl3 / 2;
        D(2, 1) = -nl5 / 2 + nl3 / 2;
        D(3, 1) = nl5 / 2 - nl7 / 2;
        D(4, 1) = nl7 / 2 - nl9 / 2;
        D(5, 1) = nl9 / 2 - nl11 / 2;
        D(1, 2) = D(2, 1);
        D(2, 2) = nl1 - nl7 / 2;
        D(3, 2) = -nl9 / 2 + nl3 / 2;
        D(4, 2) = nl5 / 2 - nl11 / 2;
        D(5, 2) = -nl13 / 2 + nl7 / 2;
        D(1, 3) = D(3, 1);
        D(2, 3) = D(3, 2);
        D(3, 3) = nl1 - nl11 / 2;
        D(4, 3) = nl3 / 2 - nl13 / 2;
        D(5, 3) = nl5 / 2;
        D(1, 4) = D(4, 1);
        D(2, 4) = D(4, 2);
        D(3, 4) = D(4, 3);
        D(4, 4) = nl1;
        D(5, 4) = nl3 / 2;
        D(1, 5) = D(5, 1);
        D(2, 5) = D(5, 2);
        D(3, 5) = D(5, 3);
        D(4, 5) = D(5, 4);
        D(5, 5) = nl1;

    case 6
        e = [e(1) 0 e(2) 0 e(3) 0 e(4) 0 e(5) 0 e(6) 0];
        % (Analytic symbolic expressions; see GetCouplKerrAnalyt for full form)
        error('nHarm=6 analytic form is large; use GetCouplKerrFFT instead');

    case 7
        e = [e(1) 0 e(2) 0 e(3) 0 e(4) 0 e(5) 0 e(6) 0 e(7) 0];
        error('nHarm=7 analytic form is large; use GetCouplKerrFFT instead');

    otherwise
        error('max nHarms = 7');
end

end
