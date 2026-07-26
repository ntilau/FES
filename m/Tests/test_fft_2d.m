%% test fft2d

N = 21;
x = ones(N,1)*linspace(0,2*pi*(N-1)/N,N);
y = x.';
f = sin(y).*sin(2*x);
% surf(f)

F = fft2(f);
figure; surf(f);
figure; surf(20*log10(fftshift(abs(F)/max(max(abs(F))))))
