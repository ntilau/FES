function [dir1model,dir2model,ldirmodel,dir1s,dir2s,ldirs,dir1b,dir2b,ldirb]=linAppSrfIGES(ParameterData,model,UV,srfind,srfs,UVs,srfinds,srfbounds,UVb,srfindb,dirb,nrmlb,tol,maxl)
% linAppSrfIGES returns local linear approximations (rectangles) at sampled points on a surfaces of an IGES-object.
% It uses the output from projpartIGES and srfRepInProjection
%
% Usage:
%
% [dir1model,dir2model,ldirmodel,dir1s,dir2s,ldirs,dir1b,dir2b,ldirb]=...
%       linAppSrfIGES(ParameterData,model,UV,srfind,srfs,UVs,srfinds,srfbounds,UVb,srfindb,dirb,nrmlb,tol,maxl)
%
% Input:
%
% See help of projpartIGES and srfRepInProjection for more information about the input variables
% tol - A positive number giving the tolerance of how much the local linear approximation (rectangles) can deviate
%       compared to a local quadratic approximation
% maxl - The maximal length of the approximating rectangles
%
% m-file can be downloaded at
% http://www.mathworks.com/matlabcentral/fileexchange/13253-iges-toolbox
%
% written by Per Bergström 2012-03-13
%


tol=max(tol,1e-14);

nummodel=size(model,2);

dir1model=zeros(3,nummodel);
dir2model=zeros(3,nummodel);
ldirmodel=zeros(2,nummodel);

for i=1:nummodel

    sind=srfind(i);
    if sind>0

        if or(ParameterData{sind}.type==128,ParameterData{sind}.type==144)

            [Ptmp,ntmp,dir1,dir2,ldir]=quadAppNrb(ParameterData{sind}.nurbs,UV(:,i),ParameterData{sind}.dnurbs,ParameterData{sind}.d2nurbs,tol);

            ldir=min(ldir,maxl);

            if ldir(1)>ldir(2)
                dir1model(:,i)=dir1;
                dir2model(:,i)=dir2;
                ldirmodel(:,i)=ldir;
            else
                dir1model(:,i)=dir2;
                dir2model(:,i)=dir1;
                ldirmodel(1,i)=ldir(2);
                ldirmodel(2,i)=ldir(1);
            end

        elseif ParameterData{sind}.type==108

            nrml=ParameterData{sind}.normal;
            nulsp=null(nrml');
            dir1model(:,i)=nulsp(:,1);
            dir2model(:,i)=nulsp(:,2);
            ldirmodel(:,i)=maxl;

        end

    end

end

nums=size(srfs,2);

dir1s=zeros(3,nums);
dir2s=zeros(3,nums);
ldirs=zeros(2,nums);

for i=1:nums

    sind=srfinds(i);

    if or(ParameterData{sind}.type==128,ParameterData{sind}.type==144)

        [Ptmp,ntmp,dir1,dir2,ldir]=quadAppNrb(ParameterData{sind}.nurbs,UVs(:,i),ParameterData{sind}.dnurbs,ParameterData{sind}.d2nurbs,tol);

        ldir=min(ldir,maxl);

        if ldir(1)>ldir(2)
            dir1s(:,i)=dir1;
            dir2s(:,i)=dir2;
            ldirs(:,i)=ldir;
        else
            dir1s(:,i)=dir2;
            dir2s(:,i)=dir1;
            ldirs(1,i)=ldir(2);
            ldirs(2,i)=ldir(1);
        end

    elseif ParameterData{sind}.type==108

        nrml=ParameterData{sind}.normal;
        nulsp=null(nrml');
        dir1s(:,i)=nulsp(:,1);
        dir2s(:,i)=nulsp(:,2);
        ldirs(:,i)=maxl;

    end

end

numb=size(srfbounds,2);

dir1b=zeros(3,numb);
dir2b=zeros(3,numb);
ldirb=zeros(2,numb);

for i=1:numb

    sind=srfindb(i);

    dir1b(:,i)=dirb(:,i);
    dir2b(:,i)=cross(dirb(:,i),nrmlb(:,i));

    if or(ParameterData{sind}.type==128,ParameterData{sind}.type==144)

        [Ptmp,ntmp,dir1,dir2,ldir]=quadAppNrb(ParameterData{sind}.nurbs,UVb(:,i),ParameterData{sind}.dnurbs,ParameterData{sind}.d2nurbs,tol);

        ldir=min(ldir,maxl);

        ldirb(1,i)=0.5*min(ldir);

    elseif ParameterData{sind}.type==108

        ldirb(1,i)=0.1*maxl;

    end

    ldirb(2,i)=0.05*ldirb(1,i);

end


function [P,normal,dir1,dir2,R,quadcoef1,quadcoef2]=quadAppNrb(nurbs,UV,dnurbs,d2nurbs,tol)

R=[0;0];

% NURBS evaluation
[P,Su,Sv,Suu,Suv,Svv]=nrbevalIGES(nurbs,UV,dnurbs,d2nurbs);

Eu=norm(Su);
Su=Su/Eu;

Ev=norm(Sv);
Sv=Sv/Ev;

normal=cross(Su,Sv);
normal=normal/norm(normal);

% quadratic function of z*normal dependent of x*Su and y*Su
% z=a*x^2+b*y^2+c*x*y

a=dot(Suu,normal)/(2*Eu^2);
b=dot(Svv,normal)/(2*Ev^2);
c=dot(Suv,normal)/(Eu*Ev);

% Orthonormal basis

So=Sv-dot(Su,Sv)*Su;

Eo=norm(So);
So=So/Eo;

transfMat=[1 -dot(Su,Sv)/Eo;0 1/Eo];

% quadratic matrix

quadMat=transfMat'*[a 0.5*c;0.5*c b]*transfMat;

[V,D]=eig(quadMat);

for j=1:2
    T=tol;
    acoef=abs(D(j,j));
    for NRiter=1:10
        T=(8*acoef*T^3+T^2+tol^2)/(12*acoef*T^2+2*T);
    end
    if abs(4*acoef*T^3+T^2-tol^2)>1e-12
        for NRiter=1:10
            T=(8*acoef*T^3+T^2+tol^2)/(12*acoef*T^2+2*T);
        end
    end
    if acoef>1e-15
        T=min(sqrt(T/acoef),1/tol);
    else
        T=1/tol;
    end
    R(j)=T+2*acoef^2*T^3;
end

dir1=[Su So]*V(:,1);
dir2=[Su So]*V(:,2);

dir1=dir1/norm(dir1);
dir2=dir2/norm(dir2);

quadcoef1=D(1,1);
quadcoef2=D(2,2);
