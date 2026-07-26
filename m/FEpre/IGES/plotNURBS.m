function plotNURBS(nurbs,clr,nu,nv)
% plotNURBS plots a NURBS entity
%
% Usage:
%
% plotNURBS(nurbs,clr,nu,nv)
%
% Input:
%
% nurbs - NURBS structure
% clr -  color of plot (optional). Default is [0.7,0.7,0.97]
%        If color is a scalar, the plot is in black and white
% nu - number of evaluation points for the parameter u (optional). Default is 20 for surfaces and 200 for curves
% nv - number of evaluation points for the parameter v (optional). Default is 20
%
% m-file can be downloaded at
% http://www.mathworks.com/matlabcentral/fileexchange/13253-iges-toolbox
%
% written by Per Bergström 2012-02-22
%

if nargin<1
    error('plotNURBS must have an input!');
end

if length(nurbs.order)>1

    % NURBS surface

    if nargin<4
        nv=20;
        if nargin<3
            nu=20;
            if nargin<2
                clr=[0.7,0.7,0.97];
            end
        end
    end

    if isempty(clr)
        clr=[0.7,0.7,0.97];
    end

    [P,UV,TRI]=nrbSrfRegularEvalIGES(nurbs,nurbs.knots{1}(1),nurbs.knots{1}(end),nu,nurbs.knots{2}(1),nurbs.knots{2}(end),nv);

    xc=nurbs.coefs(1,:,:)./nurbs.coefs(4,:,:);
    yc=nurbs.coefs(2,:,:)./nurbs.coefs(4,:,:);
    zc=nurbs.coefs(3,:,:)./nurbs.coefs(4,:,:);

    if or(ischar(clr),not(isscalar(clr)))

        patch('faces',TRI,'vertices',P','FaceColor',clr,'EdgeColor','none'), hold on

        light

        for i=1:nurbs.number(1)
            plot3(reshape(xc(:,i,:),1,[]),reshape(yc(:,i,:),1,[]),reshape(zc(:,i,:),1,[]),'k--');
        end
        for i=1:nurbs.number(2)
            plot3(xc(:,:,i),yc(:,:,i),zc(:,:,i),'k--');
        end

        plot3(xc(:),yc(:),zc(:),'r.','MarkerSize',10);

    else

        QUA=zeros((nu-1)*(nv-1),4);
        for j=1:(nv-1)
            for i=1:(nu-1)
                QUA(i+(j-1)*(nu-1),:)=[i+nu*(j-1),i+1+nu*(j-1),i+1+nu*j,i+nu*j];
            end
        end

        lw=1.5;
        patch('faces',QUA,'vertices',P','FaceColor','w','FaceAlpha',1,'EdgeColor','k','LineWidth',lw), hold on

        lw=0.5;
        for i=1:nurbs.number(1)
            plot3(reshape(xc(:,i,:),1,[]),reshape(yc(:,i,:),1,[]),reshape(zc(:,i,:),1,[]),'k--','LineWidth',lw);
        end
        for i=1:nurbs.number(2)
            plot3(xc(:,:,i),yc(:,:,i),zc(:,:,i),'k--','LineWidth',lw);
        end

        plot3(xc(:),yc(:),zc(:),'k.','MarkerSize',14);

    end

else

    % NURBS curve

    if nargin<3
        nu=200;
        if nargin<2
            clr=[0.7,0.7,0.97]*0.5;
        end
    end

    if isempty(clr)
        clr=[0.7,0.7,0.97]*0.5;
    end

    U=linspace(nurbs.knots(1),nurbs.knots(end),nu);
    P=nrbevalIGES(nurbs,U);

    xc=nurbs.coefs(1,:)./nurbs.coefs(4,:);
    yc=nurbs.coefs(2,:)./nurbs.coefs(4,:);
    zc=nurbs.coefs(3,:)./nurbs.coefs(4,:);

    lw=0.5;
    plot3(xc,yc,zc,'k--','LineWidth',lw), hold on

    if or(ischar(clr),not(isscalar(clr)))

        plot3(xc,yc,zc,'r.','MarkerSize',10);
        plot3(P(1,:),P(2,:),P(3,:),'-','Color',clr,'LineWidth',2);

    else

        plot3(xc,yc,zc,'k.','MarkerSize',14);
        plot3(P(1,:),P(2,:),P(3,:),'k-','LineWidth',2);

    end

end

axis image

sc=0.15;

xl=xlim;
dx=xl(2)-xl(1);

yl=ylim;
dy=yl(2)-yl(1);

zl=zlim;
dz=zl(2)-zl(1);

ddd=0.25*max([dx,dy,dz]);

dx=max(dx,ddd);
dy=max(dy,ddd);
dz=max(dz,ddd);

xl(1)=xl(1)-sc*dx;
xl(2)=xl(2)+sc*dx;
xlim(xl);

yl(1)=yl(1)-sc*dy;
yl(2)=yl(2)+sc*dy;
ylim(yl);

zl(1)=zl(1)-sc*dz;
zl(2)=zl(2)+sc*dz;
zlim(zl);
