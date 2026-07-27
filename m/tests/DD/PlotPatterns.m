load dataPlots
idx = find(Thetadeg == 90);
theta1 = Phideg(idx)*pi/180;
gain1 = dBGainTotal(idx);
idx1 = find(Phideg == 0);
idx2 = find(Phideg==180);
theta2 = [Thetadeg(idx1); -Thetadeg(idx2)]*pi/180;
gain2 = [dBGainTotal(idx1); dBGainTotal(idx2)];
max(gain1)

% vf_plotFFPolarCutPlanes(figure(1), 10, 1, ...
%   thetaFF, gaint, gaint, ' ' , ...
%   'prova')
ref = min(min(gain1),min(gain2));
polar(theta1, gain1-ref)
%hold on
%polar(theta2, gain2-ref)
