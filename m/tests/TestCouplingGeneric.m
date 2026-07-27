%% TestCouplingGeneric - Test generic FFT-based coupling for polynomial nonlinearity
clear all;
Config();
format longg
clear all; close all;

Sys.HBharms = 1:3;
Sys.freq = 1;

Mtrl = GetMtrlParamsFFT(Sys);

int = 1;
idx = 1:2:max(Sys.HBharms);
E = int * ones(floor(Mtrl.nHarms / 2) + 1, 1);
a = 10;
b = 5;
c = 1;

%% Compute coupling for cubic polynomial
Mtrl.func = @(x)(a + b * x + c * x .^ 2);
Field.E = int * ones(2 * Mtrl.nHarms, 1);
Field.E(2:2:end) = 0;
Field.Sin = sin(2 * pi * Sys.freq * Sys.HBharms.' * Mtrl.t);
Field.Cos = cos(2 * pi * Sys.freq * Sys.HBharms.' * Mtrl.t);
Mtrl.Sin = sin(2 * pi * Mtrl.fnl.' * Mtrl.t);
Mtrl.Cos = cos(2 * pi * Mtrl.fnl.' * Mtrl.t);

[Mtrl, Field] = GetCouplFFT(Mtrl, Field);

disp(Mtrl.D * Field.E)
