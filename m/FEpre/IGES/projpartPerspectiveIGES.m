function [model,UV,srfind,srfDerivind,srfDer,distR,pointOref]=projpartPerspectiveIGES(ParameterData,R,normal,pdir,sdir,pointO,dPerspParam,np,ns)
% projpartPerspectiveIGES returns points of perspective projections on surfaces given in an IGES-file
%
% Usage:
%
% [model,UV,srfind,srfDerivind,srfDer,distR,pointOref]=projpartPerspectiveIGES(ParameterData,...
%            R,normal,pdir,sdir,dPerspParam,np,ns)
%
% Input:
%
% ParameterData - Parameter data from IGES file. ParameterData
%                 is one of the output from IGES2MATLAB
% R - The perspective projection origin, eye position
% normal - The normal of the projection plane, which is one lengh unit from R in the direction of normal
%          The direction of normal is toward the surface
% pdir - The first (primary) direction in which projection points lies
% sdir - The second (secondary) direction in which projection points lies
% pointO - A point, !=R. Its projective projection is the reference projection
% dPerspParam - the space between the projected points in the perspective plane
% np - number of projections in primary direction
% ns - number of projections in secondary direction
%
% Output:
%
% model - points of projetions.
% UV - the parameter values for corresponding model point from original surface
% srfind - The index of surface in ParameterData for corresponding model point. srfind(i)==0 means no projection
% srfDerivind - The index of surface derivatives in srfDer for corresponding model point
% srfDer - Cell array with surface first and second derivative for all model points
% distR - distances between model points and R
% pointOref - pointO in perspective projection
%
% m-file can be downloaded at
% http://www.mathworks.com/matlabcentral/fileexchange/13253-iges-toolbox
%
% written by Per Bergström 2010-10-05
%

if nargin<9
    error('projIGES must have 9 input arguments!');
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

if size(pointO,1)<size(pointO,2)
    pointO=pointO';
end

if norm(pointO-R)<1e-3
    error('pointO too close to R!');
end

if not(and(and(mpdir==3,mnormal==3),and(msdir==3,mR==3)))
    error('Length of R, normal, pdir and sdir must be 3!');
end

if length(dPerspParam)~=1
    error('dPerspParam must be a scalar!');
elseif dPerspParam<eps
    error('dPerspParam must be larger than eps!');
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

pointOref=pointO-R;
pointOref=R+pointOref/dot(normal,pointOref);

[model,UV,srfind,srfDerivind,srfDer,nmodel,distR]=projPerspectiveIGESsub(ParameterData,normal,pdir,sdir,pointOref,dPerspParam,np,ns,R);


function [model,UV,srfind,srfDerivind,srfDer,nmodel,distR]=projPerspectiveIGESsub(ParameterData,normal,pdir,sdir,pointOref,dPerspParam,np1,np2,R)

nmodel=np1*np2;

np_1=np1-1;
ns_1=np2-1;

model=zeros(3,nmodel);

UV=zeros(2,nmodel);

srfind=zeros(1,nmodel);
srfindsup=zeros(1,nmodel);

normalz=Inf*ones(1,nmodel);

distR=-ones(1,nmodel);

% For triangulation comparison

for i=1:length(ParameterData)   % Triangulate each surface and find projection on triangulation

    [PTRI,isSCP,isSup,TRI,UV0,srfind0]=retSrfCrvPnt(1,ParameterData,1,i,200,0);

    if and(isSCP,not(isSup))

        PTRIperspective=xyz2persp(PTRI,normal,pdir,sdir,R,pointOref,dPerspParam);

        if np_1>min(PTRIperspective(1,:))
            if 0<max(PTRIperspective(1,:))
                if ns_1>min(PTRIperspective(2,:))
                    if 0<max(PTRIperspective(2,:))
                        for j=1:size(TRI,1)
                            ind1s=floor(min(PTRIperspective(1,TRI(j,:))));
                            if ind1s<np_1
                                ind1e=ceil(max(PTRIperspective(1,TRI(j,:))));
                                if ind1e>0
                                    ind2s=floor(min(PTRIperspective(2,TRI(j,:))));
                                    if ind2s<ns_1
                                        ind2e=ceil(max(PTRIperspective(2,TRI(j,:))));
                                        if ind2e>0
                                            for ii=max(0,ind1s):min(np_1,ind1e)
                                                for jj=max(0,ind2s):min(ns_1,ind2e)

                                                    dirtmp=pointOref+ii*dPerspParam*pdir+jj*dPerspParam*sdir-R;
                                                    dirtmp=dirtmp/norm(dirtmp);

                                                    vrbls=[dirtmp PTRI(:,TRI(j,1))-PTRI(:,TRI(j,2)) PTRI(:,TRI(j,1))-PTRI(:,TRI(j,3))]\(PTRI(:,TRI(j,1))-R);

                                                    if vrbls(1)>0.0 && vrbls(2)>-1e-8 && vrbls(3)>-1e-8 && (vrbls(2)+vrbls(3))<1.00000001

                                                        indtmp=ii*np2+jj+1;

                                                        if vrbls(1)<normalz(indtmp)
                                                            normalz(indtmp)=vrbls(1);
                                                            UV(:,indtmp)=UV0(:,TRI(j,1))+vrbls(2)*(UV0(:,TRI(j,2))-UV0(:,TRI(j,1)))+vrbls(3)*(UV0(:,TRI(j,3))-UV0(:,TRI(j,1)));
                                                            srfind(indtmp)=srfind0;
                                                            srfindsup(indtmp)=i;
                                                            model(:,indtmp)=R+vrbls(1)*dirtmp;
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
    clear PTRI PTRIperspective isSCP isSup TRI UV0 srfind0 vrbls
end
clear normalz
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

    for i=sti:nmodel

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
            elseif ParameterData{soi}.type==108
                srfDer{srfDerind}.type=108;
                srfDer{srfDerind}.name='PLANE ENTITY';
                srfDer{srfDerind}.supind=srfindsup(indsoP(i));
            else
                srfDer{srfDerind}.type=ParameterData{soi}.type;
                srfDer{srfDerind}.name='UNKNOWN ENTITY';
            end
        end

        srfDerivind(indsoP(i))=srfDerind;

    end

end

% For each model point. Find the projection on the corresponding surface.
for ii=0:np_1
    for jj=0:ns_1
        indtmp=ii*np2+jj+1;
        if srfind(indtmp)>0
            if srfDer{srfDerivind(indtmp)}.type==128
                dirtmp=pointOref+ii*dPerspParam*pdir+jj*dPerspParam*sdir-R;
                dirtmp=dirtmp/norm(dirtmp);
                [model(:,indtmp),UV(:,indtmp)]=closestNrbLinePointIGES(srfDer{srfDerivind(indtmp)}.nurbs,srfDer{srfDerivind(indtmp)}.dnurbs,srfDer{srfDerivind(indtmp)}.d2nurbs,UV(:,indtmp),model(:,indtmp),dirtmp);
                distR(indtmp)=norm(model(:,indtmp)-R);
            end
        end
    end
end

function PTRIperspective=xyz2persp(PTRI,normal,pdir,sdir,R,pointOref,dPerspParam)

PTRIperspective=zeros(2,size(PTRI,2));

for i=1:size(PTRI,2)
    ptmp=PTRI(:,i)-R;
    ptmp=R+ptmp/dot(normal,ptmp)-pointOref;

    PTRIperspective(1,i)=(pdir'*ptmp)/dPerspParam;
    PTRIperspective(2,i)=(sdir'*ptmp)/dPerspParam;
end
