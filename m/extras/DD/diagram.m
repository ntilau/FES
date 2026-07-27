% builds the polar diagram
function diagram(rmax,span,dSpan)
  th1 = linspace(0,2*pi,101);
  xunit = cos(th1);
  yunit = sin(th1);
  inds = 1:(length(th1)-1)/4:length(th1);
  xunit(inds(2:2:4)) = zeros(2,1);
  yunit(inds(1:2:5)) = zeros(3,1);
  patch('xdata',xunit*rmax,'ydata',yunit*rmax, ...
    'edgecolor',[0 0 0],'facecolor',[1 1 1],...
    'facealpha', 1,'LineWidth',1, 'handlevisibility','off');
  hold on;
  th2 = (1:6)*2*pi/12;
  cst = cos(th2); snt = sin(th2);
  cs = [-cst; cst];
  sn = [-snt; snt];
  line(rmax*cs,rmax*sn,'linestyle','-','color','k','linewidth',.1,...
    'handlevisibility','off');
  rt = 1.1*rmax;
  for i = 1:length(th2)
    text(rt*cst(i),rt*snt(i),[int2str((90-i*30)),...
      '°'], 'horizontalalignment','center',...
      'handlevisibility','off');
    if i == length(th2)
      loc = int2str(90);
    elseif i ~= 3
      val = - 90- sign(i-6)*i*30;
      loc = int2str( -(- sign(val)*180 + val));
    else
      loc = int2str( 180 );
    end
    text(-rt*cst(i),-rt*snt(i),[loc '°'],'horizontalalignment','center',...
    'handlevisibility','off')
  end
  hold on;
  i=1;
  unCenter = 0;
  x=(span-3)*cos(pi/2-th1); y=(span-3)*sin(pi/2-th1);
  plot(x,y,'-k','LineWidth',.1,'handlevisibility','off');
  text(x(1)+.5,y(1)+unCenter,'-3 dB','verticalalignment','bottom','FontSize',8);
  while i*dSpan<rmax
    x=(span-i*dSpan)*cos(pi/2-th1); y=(span-i*dSpan)*sin(pi/2-th1);
    plot(x,y,'-k','LineWidth',.1,'handlevisibility','off');
    text(x(1)+.5,y(1)+unCenter,num2str(-i*dSpan),'verticalalignment','bottom',...
      'FontSize',8);
    i=i+1;
  end
end
