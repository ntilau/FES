% test
clc
syms a11 a21 a22 a31 a32 a33;
A = [a11 0 0 ; a21 a22 0; a31 a32 a33];
inv(A)
