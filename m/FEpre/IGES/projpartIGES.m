function [model,UV,srfind,srfDerivind,srfDer,numpoints]=projpartIGES(ParameterData,R,normal,pdir,sdir,dp,nppos,npneg,nspos,nsneg)
% PROJPARTIGES returns points of projections on surfaces from an IGES-file.
%
% Usage:
%
% [model,UV,srfind,srfDerivind,srfDer,numpoints]=projpartIGES(ParameterData,...
%            R,normal,pdir,sdir,dp,nppos,npneg,nspos,nsneg)
%
% Input:
%
% ParameterData - Parameter data from IGES file. ParameterData
%                 is one of the output from IGES2MATLAB.
% R - The projection origo. A point on the surface where the origo of
%     projections is located.
% normal - The projection normal. The direction of normal is toward the surface.
% pdir - The first (primary) direction in which projection points lies.
% sdir - The second (secondary) direction in which projection points lies.
% dp - the distance between the projected points.
% nppos - number of projections in positive primary direction.
% npneg - number of projections in negative primary direction.
% nspos - number of projections in positive secondary direction.
% nsneg - number of projections in negative secondary direction.
%
% Output:
%
% model - points of projetions.
% UV - the parameter values for corresponding model point from original surface.
% srfind - The index of surface in ParameterData for corresponding model point. srfind(i)==0 means no projection.
% srfDerivind - The index of surface derivatives in srfDer for corresponding model point.
% srfDer - Cell array with surface first and second derivative for all model points.
%
% m-file can be downloaded for free at
% http://www.mathworks.com/matlabcentral/fileexchange/13253-iges-toolbox
%
% written by Per Bergström 2012-02-22
%

if nargin<10
    error('projIGES must have 10 input arguments!');
end

if not(iscell(ParameterData))
    error('ParameterData must be a cell array!');
end

[mR,nR]=size(R);

if mR<nR
    R=R';
    [mR,nR]=size(R);
end

[mnormal,nnormal]=size(normal);

if mnormal<nnormal
    normal=normal';
    [mnormal,nnormal]=size(normal);
end

[mpdir,npdir]=size(pdir);

if mpdir<npdir
    pdir=pdir';
    [mpdir,npdir]=size(pdir);
end

[msdir,nsdir]=size(sdir);

if msdir<nsdir
    sdir=sdir';
    [msdir,nsdir]=size(sdir);
end

if not(and(and(mpdir==3,mnormal==3),and(msdir==3,mR==3)))
    error('Length of R, normal, pdir and sdir must be 3!');
end

if length(dp)~=1
    error('dp must be a scalar!');
elseif dp<eps
    error('dp must be larger than eps!');
end

if or(isempty(nppos),length(nppos)>1)
    nppos=0;
elseif nppos<0
    nppos=0;
else
    nppos=round(nppos);
end

if or(isempty(npneg),length(npneg)>1)
    npneg=0;
elseif npneg<0
    npneg=0;
else
    npneg=round(npneg);
end

if or(isempty(nspos),length(nspos)>1)
    nspos=0;
elseif nspos<0
    nspos=0;
else
    nspos=round(nspos);
end

if or(isempty(nsneg),length(nsneg)>1)
    nsneg=0;
elseif nsneg<0
    nsneg=0;
else
    nsneg=round(nsneg);
end

normal=normal/norm(normal);

pdir=pdir-dot(pdir,normal)*normal;   % primary direction

nopd=norm(pdir);
if nopd<1e-6
    error('pdir can not be parallel to normal');
else
    pdir=pdir/nopd;                  % orthogonal to normal
end

sdirdir=cross(normal,pdir);
sdirdir=sdirdir/norm(sdirdir);

sdir=dot(sdir,sdirdir)*sdirdir;      % secondary direction

nosd=norm(sdir);
if nosd<1e-6
    error('Illegeal pdir, sdir or normal!');
else
    sdir=sdir/nosd;
end

numpoints=[(nppos+npneg+1),(nspos+nsneg+1)];

pO=[pdir sdir]\(R-pdir*(npneg*dp)-sdir*(nsneg*dp));

[model,UV,srfind,srfDerivind,srfDer,nmodel]=projIGESsub(ParameterData,normal,pdir,sdir,dp,nppos+npneg+1,nspos+nsneg+1,pO);


function [model,UV,srfind,srfDerivind,srfDer,nmodel]=projIGESsub(ParameterData,normal,pdir,sdir,dp,np1,np2,pO)

nmodel=np1*np2;

model=zeros(3,nmodel);

UV=zeros(2,nmodel);

srfind=zeros(1,nmodel);
srfindsup=zeros(1,nmodel);

normalz=Inf*ones(1,nmodel);

p=[0;0];
ab=[0;0];
Ptemp=[0;0;0];
coordmat=[0 0;0 0];

% For triangulation comparison

for i=1:length(ParameterData)   % Triangulate each surface and find projection on triangulation

    [PTRI,isSCP,isSup,TRI,UV0,srfind0]=retSrfCrvPnt(1,ParameterData,1,i,200,0);

    if and(isSCP,not(isSup))

        pcoord=pdir'*PTRI;

        if (pO(1)+(np1-1)*dp)>min(pcoord)
            if pO(1)<max(pcoord)
                scoord=sdir'*PTRI;
                if (pO(2)+(np2-1)*dp)>min(scoord)
                    if pO(2)<max(scoord)

                        for j=1:size(TRI,1)

                            ind1s=floor((min(pcoord(TRI(j,:)))-pO(1))/dp);
                            if ind1s<np1
                                ind1e=ceil((max(pcoord(TRI(j,:)))-pO(1))/dp);
                                if ind1e>-1
                                    ind2s=floor((min(scoord(TRI(j,:)))-pO(2))/dp);
                                    if ind2s<np2
                                        ind2e=ceil((max(scoord(TRI(j,:)))-pO(2))/dp);
                                        if ind2e>-1

                                            coordmat(:)=[scoord(TRI(j,2))-scoord(TRI(j,3)),scoord(TRI(j,3))-scoord(TRI(j,1)),pcoord(TRI(j,3))-pcoord(TRI(j,2)),pcoord(TRI(j,1))-pcoord(TRI(j,3))];
                                            c=det(coordmat);

                                            if abs(c)>1e-12

                                                coordmat=coordmat/c;

                                                for ii=max(0,ind1s):min(np1-1,ind1e)

                                                    p(1)=pO(1)+ii*dp-pcoord(TRI(j,3));

                                                    for jj=max(0,ind2s):min(np2-1,ind2e)

                                                        p(2)=pO(2)+jj*dp-scoord(TRI(j,3));

                                                        ab(:)=coordmat*p;
                                                        c=1-sum(ab);

                                                        if ab(1)>-1e-8 && ab(2)>-1e-8 && c>-1e-8

                                                            Ptemp(:)=ab(1)*PTRI(:,TRI(j,1))+ab(2)*PTRI(:,TRI(j,2))+c*PTRI(:,TRI(j,3));
                                                            normalztemp=dot(Ptemp,normal);

                                                            if normalztemp<normalz(ii*np2+jj+1)
                                                                normalz(ii*np2+jj+1)=normalztemp;
                                                                UV(:,ii*np2+jj+1)=ab(1)*UV0(:,TRI(j,1))+ab(2)*UV0(:,TRI(j,2))+c*UV0(:,TRI(j,3));
                                                                srfind(ii*np2+jj+1)=srfind0;
                                                                srfindsup(ii*np2+jj+1)=i;
                                                                model(:,ii*np2+jj+1)=Ptemp;
                                                            end

                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    clear PTRI isSCP isSup TRI UV0 srfind0 pcoord scoord srfind0;
end
clear normalz normalztemp p ab c Ptemp coordmat
clear functions

[sosrfind,indsoP]=sort(srfind);

numbder=0;
soitmp=nmodel+1;
testFlag=true;
for i=1:nmodel
    if testFlag
        if sosrfind(i)>0
            sti=i;
            soitmp=sosrfind(i);
            soi=soitmp-1;
            numbder=1;
            testFlag=false;
        end
    elseif sosrfind(i)>soitmp
        soitmp=sosrfind(i);
        numbder=numbder+1;
    end
end
clear soitmp

srfDer=cell(1,numbder);
srfDerivind=zeros(1,nmodel);

if numbder>0

    srfDerind=1;

    for i=sti:nmodel           % For each model point. Find the projection on the corresponding surface.

        if sosrfind(i)>soi
            soi=sosrfind(i);
            srfDerind=srfDerind+1;
            if or(ParameterData{soi}.type==128,ParameterData{soi}.type==144)
                srfDer{srfDerind}.type=128;
                srfDer{srfDerind}.name='RATIONAL B-SPLINE SURFACE ENTITY';
                srfDer{srfDerind}.nurbs=ParameterData{soi}.nurbs;
                srfDer{srfDerind}.dnurbs=ParameterData{soi}.dnurbs;
                srfDer{srfDerind}.d2nurbs=ParameterData{soi}.d2nurbs;
                srfDer{srfDerind}.supind=srfindsup(indsoP(i));
                testFlag=true;
            elseif ParameterData{soi}.type==108
                srfDer{srfDerind}.type=108;
                srfDer{srfDerind}.name='PLANE ENTITY';
                srfDer{srfDerind}.supind=srfindsup(indsoP(i));
                testFlag=false;
            else
                srfDer{srfDerind}.type=ParameterData{soi}.type;
                srfDer{srfDerind}.name='UNKNOWN ENTITY';
                testFlag=false;
            end
        end

        srfDerivind(indsoP(i))=srfDerind;
        if testFlag
            [model(:,indsoP(i)),UV(:,indsoP(i))]=closestNrbLinePointIGES(srfDer{srfDerind}.nurbs,srfDer{srfDerind}.dnurbs,srfDer{srfDerind}.d2nurbs,UV(:,indsoP(i)),model(:,indsoP(i)),normal);
        end

    end

end
