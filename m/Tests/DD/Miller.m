% Test Miller
A = [1 2; 4 5];
B = [-1 3; 0 0];
% det(A)
% det(A-B)
inv(A-B)
inv(A)+ 1/(1-trace(B*inv(A)))*inv(A)*B*inv(A)
%%
A = [1 2; 4 5];
B = [-1 3; 4 20];
% det(A)
% det(A+B)
B1 = B*0;
B2 = B*0;
B1(1,:) = B(1,:);
B2(2,:) = B(2,:);
inv(A-B)

C1 = inv(A) + 1/(1-trace(B1*inv(A)))*inv(A)*B1*inv(A);
C2 = C1 + 1/(1-trace(B2*C1))*C1*B2*C1
