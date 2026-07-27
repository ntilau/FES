function X = SolvDir(Sys)
% Solve the full (direct) system A * X = B.
% X = A \ B  (sparse direct solve)

X = Sys.A \ Sys.B;

end
