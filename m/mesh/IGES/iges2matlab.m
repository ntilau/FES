function [ParameterData,EntityType,numEntityType,unknownEntityType,numunknownEntityType]=iges2matlab(igsfile,showlines,useTransformationEnityArb)
% IGES2MATLAB extracts the parameter data in an IGES-file to MATLAB format
%
% Usage:
%
% [ParameterData,EntityType,numEntityType,unknownEntityType,numunknownEntityType]=...
%         iges2matlab(igsfile,showlines,useTransformationEnityArb)
%
% Input:
%
% igsfile - IGES file
% showlines - information flag to plotIGES (optional)
%             0 (default) show only lines in plotIGES corresponding to surfaces
%             1 show all lines in plotIGES
% useTransformationEnityArb - information flag (optional)
%             0 The transformation entity 124 are only used on the next subsequent circular/conic arc entity (100/104)
%             1 (default) The transformation entity 124 are used on the next subsequent entity
%
% Output:
%
% ParameterData - cell array with Parameter Data from igsfile
% EntityType - vector with entities in igsfile converted to matlab
% numEntityType - vector with number of entities belonging to EntityType
% unknownEntityType - vector with unknown entities for iges2matlab
% numunknownEntityType - vector with number of unknown entities
%                        belonging to unknownEntityType
%
% For entity type 126 and 128, ParameterData do also contain a nurbs
% representation. The nurbs representation is the same as in the NURBS toolbox,
% http://www.mathworks.com/matlabcentral/fileexchange/14247-nurbs
%
% ParameterData do also contain other useful information for usage in other
% functions. For curves the length is given as a parameter in ParameterData
% superior is another parameter for curves and surfaces. For curves superior=1
% means that they are defined in the parameter space for a surface. superior=0
% means that they are defined in the 3D-space. For surfaces superior=1 means
% that their domain is limited by one or more closed curves. superior=0 means
% that their domain is the domain given in the ParameterData
%
% For other parameters see the IGES specificaton at
% www.uspro.org/documents/IGES5-3_forDownload.pdf
%
% All pointers in ParameterData points to the index
% in the cell-array ParameterData.
%
% This version can not handle all possible IGES entities
%
% Example:
%
% [ParameterData,EntityType,numEntityType,unknownEntityType]=iges2matlab('example.igs');
%
% will read the parameter data in example.igs into Matlab.
%
% m-file can be downloaded at
% http://www.mathworks.com/matlabcentral/fileexchange/13253-iges-toolbox
%
% written by Per Bergström 2012-02-27
%


if nargin<3
    useTransformationEnityArb=1;
    if nargin<2
        showlines=0;
    end
end

if isempty(showlines)
    showlines=0;
end
if not(or(showlines==0,showlines==1))
    showlines=0;
end
if isempty(useTransformationEnityArb)
    useTransformationEnityArb=1;
end
if not(or(useTransformationEnityArb==0,useTransformationEnityArb==1))
    useTransformationEnityArb=1;
end

[fid,msg]=fopen(igsfile);
if fid==-1
    error(msg);
end
c = fread(fid,'uint8=>uint8')';
fclose(fid);

nwro=sum((c((81:82))==10))+sum((c((81:82))==13));
edfi=nwro-sum(c(((end-1):end))==10)-sum(c(((end-1):end))==13);
siz=length(c);
ro=round((siz+edfi)/(80+nwro));
if rem((siz+edfi),(80+nwro))~=0
    error('Input file must be an IGES-file!');
end

roind=1:ro;
SGDPT=c(roind*(80+nwro)-7-nwro);

Sfind=SGDPT==83;
Gfind=SGDPT==71;
Dfind=SGDPT==68;
Pfind=SGDPT==80;
Tfind=SGDPT==84;

sumSfind=sum(Sfind);
sumGfind=sum(Gfind);
sumDfind=sum(Dfind);
sumPfind=sum(Pfind);
sumTfind=sum(Tfind);

%------S Line information (The initial line to get things started)---------
for i=roind(Sfind)
    disp(char(c(((i-1)*(80+nwro)+1):(i*(80+nwro)-8-nwro))));
end

%---------------G Line information  (Header infomation)--------------------

G=cell(1,25);
Gstr=zeros(1,72*sumGfind);
j=1;
for i=roind(Gfind)
    Gstr(((j-1)*72+1):(j*72))=c(((i-1)*(80+nwro)+1):(i*(80+nwro)-8-nwro));
    j=j+1;
end

if and(Gstr(1)==49,Gstr(2)==72)
    G{1}=Gstr(3);
    st=4;
else
    G{1}=44;
    st=1;
end

if and(Gstr(st+1)==49,Gstr(st+2)==72)
    G{2}=Gstr(st+3);
    st=st+4;
else
    G{2}=59;
    st=st+1;
end

le=length(Gstr);
for i=3:25
    for j=(st+1):le
        if or(Gstr(j)==G{1},Gstr(j)==G{2})
            break
        end
    end
    G{i}=Gstr((st+1):(j-1));
    st=j;
end

for i=[3 4 5 6 12 15 18 21 22 25]   %string
    stind=1;
    for j=1:length(G{i})
        if G{i}(j)~=32
            stind=j;
            break
        end
    end
    for j=stind:length(G{i})
        if G{i}(j)==72
            stind=j+1;
            break
        end
    end
    endind=length(G{i});
    for j=length(G{i}):-1:1
        if G{i}(j)~=32
            endind=j;
            break
        end
    end
    G{i}=G{i}(stind:endind);
end

for i=[7 8 9 10 11 13 14 16 17 19 20 23 24]   %num
    G{i}=str2num(char(G{i}));
end

%--D Line information (Data information) & P Line information (All data)---

noent=round(sumDfind/2);
ParameterData=cell(1,noent);
roP=sumSfind+sumGfind+sumDfind;

entty=zeros(1,520);
entunk=zeros(1,520);

entiall=0;

useTrnsfrmEntity=false;
rotMat=eye(3);
transVec=zeros(3,1);

for i=(sumSfind+sumGfind+1):2:(sumSfind+sumGfind+sumDfind-1)

    entiall=entiall+1;
    Dstr=c(((i-1)*(80+nwro)+1):(i*(80+nwro)-8-nwro));
    type=str2num(char(Dstr(1:8)));
    ParameterData{entiall}.type=type;
    Pstart=str2num(char(Dstr(9:16)))+roP;

    if i==roP-1
        Pend=ro-sumTfind;
    else
        Pend=str2num(char(c(((i+1)*(80+nwro)+9):((i+1)*(80+nwro)+16))))+roP-1;
    end

    Pstr=zeros(1,64*(Pend-Pstart+1));
    j=1;
    for k=Pstart:Pend
        Pstr(((j-1)*64+1):(j*64))=c(((k-1)*(80+nwro)+1):(k*(80+nwro)-16-nwro));
        j=j+1;
    end

    Pstr(Pstr==G{1})=44;
    Pstr(Pstr==G{2})=59;

    Pvec=str2num(char(Pstr));

    % Store the entities

    % SURFACES

    if type==128

        entty(type)=entty(type)+1;

        A=1+Pvec(2)+Pvec(4);
        B=1+Pvec(3)+Pvec(5);
        C=(Pvec(2)+1)*(Pvec(3)+1);

        ParameterData{entiall}.name='B-NURBS SRF';
        ParameterData{entiall}.original=1;

        ParameterData{entiall}.superior=0;

        ParameterData{entiall}.k1=Pvec(2);
        ParameterData{entiall}.k2=Pvec(3);
        ParameterData{entiall}.m1=Pvec(4);
        ParameterData{entiall}.m2=Pvec(5);

        ParameterData{entiall}.prop1=Pvec(6);
        ParameterData{entiall}.prop2=Pvec(7);
        ParameterData{entiall}.prop3=Pvec(8);
        ParameterData{entiall}.prop4=Pvec(10);
        ParameterData{entiall}.prop5=Pvec(11);

        ParameterData{entiall}.s=Pvec(11:(11+A));
        ParameterData{entiall}.t=Pvec((12+A):(12+A+B));

        ParameterData{entiall}.w=reshape(Pvec((13+A+B):(12+A+B+C)),Pvec(2)+1,Pvec(3)+1);
        ParameterData{entiall}.p=reshape(Pvec((13+A+B+C):(12+A+B+4*C)),3,Pvec(2)+1,Pvec(3)+1);

        if useTransformationEnityArb
            if useTrnsfrmEntity
                for scndInd=1:(Pvec(3)+1)
                    ParameterData{entiall}.p(:,:,scndInd)=rotMat*(ParameterData{entiall}.p(:,:,scndInd))+repmat(transVec,1,Pvec(2)+1);
                end
                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end
        end

        ParameterData{entiall}.u=zeros(1,2);
        ParameterData{entiall}.u(1)=Pvec(13+A+B+4*C);
        ParameterData{entiall}.u(2)=Pvec(14+A+B+4*C);

        ParameterData{entiall}.v=zeros(1,2);
        ParameterData{entiall}.v(1)=Pvec(15+A+B+4*C);
        ParameterData{entiall}.v(2)=Pvec(16+A+B+4*C);

        % NURBS surface

        ParameterData{entiall}.nurbs.form='B-NURBS';

        ParameterData{entiall}.nurbs.dim=4;

        ParameterData{entiall}.nurbs.number=zeros(1,2);
        ParameterData{entiall}.nurbs.number(1)=Pvec(2)+1;
        ParameterData{entiall}.nurbs.number(2)=Pvec(3)+1;

        ParameterData{entiall}.nurbs.coefs=zeros(4,Pvec(2)+1,Pvec(3)+1);
        ParameterData{entiall}.nurbs.coefs(4,:,:)=reshape(Pvec((13+A+B):(12+A+B+C)),Pvec(2)+1,Pvec(3)+1);
        ParameterData{entiall}.nurbs.coefs(1:3,:,:)=ParameterData{entiall}.p;

        ParameterData{entiall}.nurbs.coefs(1:3,:,:)=repmat(ParameterData{entiall}.nurbs.coefs(4,:,:),3,1).*ParameterData{entiall}.nurbs.coefs(1:3,:,:);

        ParameterData{entiall}.nurbs.knots=cell(1,2);
        ParameterData{entiall}.nurbs.knots{1}=Pvec(11:(11+A));
        ParameterData{entiall}.nurbs.knots{2}=Pvec((12+A):(12+A+B));

        ParameterData{entiall}.nurbs.order=zeros(1,2);
        ParameterData{entiall}.nurbs.order(1)=Pvec(4)+1;
        ParameterData{entiall}.nurbs.order(2)=Pvec(5)+1;

        [ParameterData{entiall}.dnurbs,ParameterData{entiall}.d2nurbs]=nrbDerivativesIGES(ParameterData{entiall}.nurbs);

        ParameterData{entiall}.well=true;

    elseif type==120

        ParameterData{entiall}.name='SURFACE OF REVOLUTION';

        ParameterData{entiall}.l=round((Pvec(2)+1)/2);
        ParameterData{entiall}.c=round((Pvec(3)+1)/2);
        ParameterData{entiall}.sa=Pvec(4);
        ParameterData{entiall}.ta=Pvec(5);

        if ParameterData{ParameterData{entiall}.c}.type==110

            CRVk=1;
            CRVm=1;
            CRVt=[0 0 1 1];
            CRVw=[1 1];
            CRVp=[ParameterData{ParameterData{entiall}.c}.p1 ParameterData{ParameterData{entiall}.c}.p2];
            CRVv=[0 1];

            boool=true;

        elseif ParameterData{ParameterData{entiall}.c}.type==126

            CRVk=ParameterData{ParameterData{entiall}.c}.k;
            CRVm=ParameterData{ParameterData{entiall}.c}.m;
            CRVt=ParameterData{ParameterData{entiall}.c}.t;
            CRVw=ParameterData{ParameterData{entiall}.c}.w;
            CRVp=ParameterData{ParameterData{entiall}.c}.p;
            CRVv=ParameterData{ParameterData{entiall}.c}.v;

            boool=true;

        else

            disp(['Warning: Could not handle entity type 120 correctly in ',igsfile,'.']);
            boool=false;

        end

        if boool

            entty(128)=entty(128)+1;

            p1=ParameterData{ParameterData{entiall}.l}.p1;

            rotDir=ParameterData{ParameterData{entiall}.l}.p2-p1;
            cpDirs=CRVp;
            cpDirs(1,:)=cpDirs(1,:)-p1(1);
            cpDirs(2,:)=cpDirs(2,:)-p1(2);
            cpDirs(3,:)=cpDirs(3,:)-p1(3);
            cpDirs=[0 -rotDir(3) rotDir(2);rotDir(3) 0 -rotDir(1);-rotDir(2) rotDir(1) 0]*cpDirs;
            lDist=sum(cpDirs.^2,1);
            [sqDist,maInd]=max(lDist);

            A=[p1 ParameterData{ParameterData{entiall}.l}.p2 p1+cpDirs(:,maInd)];
            B=zeros(3,3);
            B(3,2)=norm(rotDir);
            B(2,3)=sqrt(sqDist);

            meA=mean(A,2);
            meB=mean(B,2);

            [U,Sigm,V]=svd((B-repmat(meB,1,3))*(A-repmat(meA,1,3))');
            R=U*V';
            detR=det(R);
            if detR<0
                V(:,3)=-V(:,3);
                R=U*V';
            end
            T=meB-R*meA;

            CRVp=R*(CRVp)+repmat(T,1,CRVk+1);

            vmin=Pvec(4);
            vmax=Pvec(5);

            ParameterData{entiall}.type=128;

            ParameterData{entiall}.name='B-NURBS SRF';
            ParameterData{entiall}.original=0;
            ParameterData{entiall}.previous_type=120;
            ParameterData{entiall}.previous_name='SURFACE OF REVOLUTION';

            ParameterData{entiall}.superior=0;

            ParameterData{entiall}.k1=CRVk;
            ParameterData{entiall}.k2=6;

            ParameterData{entiall}.m1=CRVm;
            ParameterData{entiall}.m2=2;

            ParameterData{entiall}.prop1=0;
            ParameterData{entiall}.prop2=0;
            ParameterData{entiall}.prop3=0;
            ParameterData{entiall}.prop4=0;
            ParameterData{entiall}.prop5=0;

            ParameterData{entiall}.s=CRVt;
            ParameterData{entiall}.t=[0 0 0 1/3 1/3 2/3 2/3 1 1 1]*(vmax-vmin)+vmin;

            wodd=cos((vmax-vmin)/6);

            ParameterData{entiall}.w=[CRVw;wodd*(CRVw);CRVw;wodd*(CRVw);CRVw;wodd*(CRVw);CRVw]';
            ParameterData{entiall}.p=zeros(3,CRVk+1,7);

            betavec=linspace(vmin,vmax,7);

            eve=true;

            for ii=1:7

                if eve
                    cob=cos(betavec(ii));
                    sib=sin(betavec(ii));
                    eve=false;
                else
                    cob=cos(betavec(ii))/wodd;
                    sib=sin(betavec(ii))/wodd;
                    eve=true;
                end

                ParameterData{entiall}.p(1,:,ii)=cob*CRVp(1,:)-sib*CRVp(2,:)-T(1);
                ParameterData{entiall}.p(2,:,ii)=sib*CRVp(1,:)+cob*CRVp(2,:)-T(2);
                ParameterData{entiall}.p(3,:,ii)=CRVp(3,:)-T(3);
                ParameterData{entiall}.p(:,:,ii)=R'*ParameterData{entiall}.p(:,:,ii);

            end

            if useTransformationEnityArb
                if useTrnsfrmEntity
                    for scndInd=1:7
                        ParameterData{entiall}.p(:,:,scndInd)=rotMat*(ParameterData{entiall}.p(:,:,scndInd))+repmat(transVec,1,Pvec(2)+1);
                    end
                    useTrnsfrmEntity=false;
                    rotMat=eye(3);
                    transVec=zeros(3,1);
                end
            end

            ParameterData{entiall}.u=CRVv;
            ParameterData{entiall}.v=[vmin vmax];


            % NURBS surface

            ParameterData{entiall}.nurbs.form='B-NURBS';
            ParameterData{entiall}.nurbs.dim=4;
            ParameterData{entiall}.nurbs.number=[ParameterData{entiall}.k1+1 ParameterData{entiall}.k2+1];
            ParameterData{entiall}.nurbs.coefs=zeros(4,CRVk+1,7);

            for ii=1:7
                ParameterData{entiall}.nurbs.coefs(4,:,ii)=ParameterData{entiall}.w(:,ii)';

                ParameterData{entiall}.nurbs.coefs(1,:,ii)=ParameterData{entiall}.nurbs.coefs(4,:,ii).*ParameterData{entiall}.p(1,:,ii);
                ParameterData{entiall}.nurbs.coefs(2,:,ii)=ParameterData{entiall}.nurbs.coefs(4,:,ii).*ParameterData{entiall}.p(2,:,ii);
                ParameterData{entiall}.nurbs.coefs(3,:,ii)=ParameterData{entiall}.nurbs.coefs(4,:,ii).*ParameterData{entiall}.p(3,:,ii);
            end

            ParameterData{entiall}.nurbs.knots=cell(1,2);
            ParameterData{entiall}.nurbs.knots{1}=ParameterData{entiall}.s;
            ParameterData{entiall}.nurbs.knots{2}=ParameterData{entiall}.t;

            ParameterData{entiall}.nurbs.order=[ParameterData{entiall}.m1+1 ParameterData{entiall}.m2+1];

            [ParameterData{entiall}.dnurbs,ParameterData{entiall}.d2nurbs]=nrbDerivativesIGES(ParameterData{entiall}.nurbs);

            ParameterData{entiall}.well=true;

        else
            entty(120)=entty(120)+1;

            ParameterData{entiall}.name='Unknown type!';
            entunk(type)=entunk(type)+1;

            ParameterData{entiall}.original=1;

            ParameterData{entiall}.length=0.00000000001;

            ParameterData{entiall}.well=false;
        end

    elseif type==144

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='TRIMMED SURFACE';

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.pts=round((Pvec(2)+1)/2);

        ParameterData{entiall}.n1=Pvec(3);
        ParameterData{entiall}.n2=Pvec(4);

        if Pvec(5)~=0
            ParameterData{entiall}.pto=round((Pvec(5)+1)/2);
        else
            ParameterData{entiall}.pto=0;
        end

        ParameterData{entiall}.pti=round((Pvec(6:(5+Pvec(4)))+1)/2);

        ParameterData{entiall}.well=true;

    elseif type==108

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='PLANE';
        ParameterData{entiall}.original=1;

        ParameterData{entiall}.superior=0;

        ParameterData{entiall}.a=Pvec(2);
        ParameterData{entiall}.b=Pvec(3);
        ParameterData{entiall}.c=Pvec(4);
        ParameterData{entiall}.d=Pvec(5);

        if Pvec(6)==0
            ParameterData{entiall}.ptr=0;
        else
            ParameterData{entiall}.ptr=round((Pvec(6)+1)/2);
        end

        ParameterData{entiall}.x=Pvec(7);
        ParameterData{entiall}.y=Pvec(8);
        ParameterData{entiall}.z=Pvec(9);
        ParameterData{entiall}.size=Pvec(10);

        ParameterData{entiall}.normal=[Pvec(2);Pvec(3);Pvec(4)];

        if useTransformationEnityArb
            if useTrnsfrmEntity
                ParameterData{entiall}.normal=rotMat*ParameterData{entiall}.normal;
                ParameterData{entiall}.a=ParameterData{entiall}.normal(1);
                ParameterData{entiall}.b=ParameterData{entiall}.normal(2);
                ParameterData{entiall}.c=ParameterData{entiall}.normal(3);
                ParameterData{entiall}.d=ParameterData{entiall}.d+dot(ParameterData{entiall}.normal,transVec);
                X=rotMat*[Pvec(7);Pvec(8);Pvec(9)]+transVec;
                ParameterData{entiall}.x=X(1);
                ParameterData{entiall}.y=X(2);
                ParameterData{entiall}.z=X(3);

                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end
        end

        ParameterData{entiall}.well=true;

        % CURVES

    elseif type==126

        entty(type)=entty(type)+1;

        N=1+Pvec(2)-Pvec(3);
        A=1+Pvec(2)+Pvec(3);

        ParameterData{entiall}.name='B-NURBS CRV';
        ParameterData{entiall}.original=1;

        if showlines
            ParameterData{entiall}.superior=0;
        else
            ParameterData{entiall}.superior=1;
        end

        ParameterData{entiall}.k=Pvec(2);
        ParameterData{entiall}.m=Pvec(3);

        ParameterData{entiall}.prop1=Pvec(4);
        ParameterData{entiall}.prop2=Pvec(5);
        ParameterData{entiall}.prop3=Pvec(6);
        ParameterData{entiall}.prop4=Pvec(7);

        ParameterData{entiall}.t=Pvec(8:(8+A));
        ParameterData{entiall}.w=Pvec((9+A):(9+A+Pvec(2)));
        ParameterData{entiall}.p=reshape(Pvec((10+A+Pvec(2)):(12+A+4*Pvec(2))),3,Pvec(2)+1);

        if useTransformationEnityArb
            if useTrnsfrmEntity
                ParameterData{entiall}.p=rotMat*(ParameterData{entiall}.p)+repmat(transVec,1,Pvec(2)+1);
                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end
        end

        ParameterData{entiall}.v=zeros(1,2);
        ParameterData{entiall}.v(1)=Pvec(13+A+4*Pvec(2));
        ParameterData{entiall}.v(2)=Pvec(14+A+4*Pvec(2));

        if Pvec(4)
            ParameterData{entiall}.xnorm=Pvec(15+A+4*Pvec(2));
            ParameterData{entiall}.ynorm=Pvec(16+A+4*Pvec(2));
            ParameterData{entiall}.znorm=Pvec(17+A+4*Pvec(2));
        else
            ParameterData{entiall}.xnorm=0;
            ParameterData{entiall}.ynorm=0;
            ParameterData{entiall}.znorm=0;
        end

        % NURBS curve

        ParameterData{entiall}.nurbs.form='B-NURBS';

        ParameterData{entiall}.nurbs.dim=4;

        ParameterData{entiall}.nurbs.number=Pvec(2)+1;

        ParameterData{entiall}.nurbs.coefs=zeros(4,Pvec(2)+1);
        ParameterData{entiall}.nurbs.coefs(4,:)=Pvec((9+A):(9+A+Pvec(2)));

        ParameterData{entiall}.nurbs.coefs(1:3,:)=ParameterData{entiall}.p;

        ParameterData{entiall}.nurbs.coefs(1:3,:)=repmat(ParameterData{entiall}.nurbs.coefs(4,:),3,1).*ParameterData{entiall}.nurbs.coefs(1:3,:);

        ParameterData{entiall}.nurbs.order=Pvec(3)+1;

        ParameterData{entiall}.nurbs.knots=Pvec(8:(8+A));

        [ParameterData{entiall}.dnurbs,ParameterData{entiall}.d2nurbs]=nrbDerivativesIGES(ParameterData{entiall}.nurbs);

        nup=500;
        p = nrbevalIGES(ParameterData{entiall}.nurbs,linspace(ParameterData{entiall}.v(1),ParameterData{entiall}.v(2),nup));
        len=sum(sqrt(sum((p(:,1:(nup-1))-p(:,2:nup)).^2,1)));
        if norm(p(:,1)-p(:,nup))<1e-3
            ParameterData{entiall}.length=3*len;
        else
            ParameterData{entiall}.length=min((len/norm(p(:,1)-p(:,nup))-1)*10+1,3)*len;
        end

        ParameterData{entiall}.well=true;

        clear nup p len N A

    elseif type==100

        ParameterData{entiall}.type=126;

        entty(126)=entty(126)+1;

        zt=Pvec(2);
        x1=Pvec(3);
        y1=Pvec(4);
        x2=Pvec(5);
        y2=Pvec(6);
        x3=Pvec(7);
        y3=Pvec(8);

        R=0.5*(sqrt((x2-x1)^2+(y2-y1)^2)+sqrt((x3-x1)^2+(y3-y1)^2));

        if ((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))<=0
            beta=2*pi-acos(((x2-x1)*(x3-x1)+(y2-y1)*(y3-y1))/(R^2));
        else
            beta=acos(((x2-x1)*(x3-x1)+(y2-y1)*(y3-y1))/(R^2));
        end

        if beta<1e-12
            beta=2*pi;
        end

        wodd=cos(beta/6);

        P0=[x2;y2];

        P1=[cos(beta/6) -sin(beta/6);sin(beta/6) cos(beta/6)]*[x2-x1;y2-y1]/wodd+[x1;y1];

        P2=[cos(beta/3) -sin(beta/3);sin(beta/3) cos(beta/3)]*[x2-x1;y2-y1]+[x1;y1];

        P3=[cos(beta/2) -sin(beta/2);sin(beta/2) cos(beta/2)]*[x2-x1;y2-y1]/wodd+[x1;y1];

        P4=[cos(2*beta/3) -sin(2*beta/3);sin(2*beta/3) cos(2*beta/3)]*[x2-x1;y2-y1]+[x1;y1];

        P5=[cos(5*beta/6) -sin(5*beta/6);sin(5*beta/6) cos(5*beta/6)]*[x2-x1;y2-y1]/wodd+[x1;y1];

        P6=[x3;y3];

        PP=[P0 P1 P2 P3 P4 P5 P6;zt*ones(1,7)];

        if useTrnsfrmEntity
            PP=rotMat*PP+repmat(transVec,1,7);
            useTrnsfrmEntity=false;
            rotMat=eye(3);
            transVec=zeros(3,1);
        end

        ParameterData{entiall}.name='B-NURBS CRV';
        ParameterData{entiall}.original=0;
        ParameterData{entiall}.previous_type=100;
        ParameterData{entiall}.previous_name='CIRCULAR ARC';

        ParameterData{entiall}.zt=zt;
        ParameterData{entiall}.x1=x1;
        ParameterData{entiall}.y1=y1;
        ParameterData{entiall}.x2=x2;
        ParameterData{entiall}.y2=y2;
        ParameterData{entiall}.x3=x3;
        ParameterData{entiall}.y3=y3;

        if showlines
            ParameterData{entiall}.superior=0;
        else
            ParameterData{entiall}.superior=1;
        end

        ParameterData{entiall}.k=6;
        ParameterData{entiall}.m=2;

        ParameterData{entiall}.prop1=1;
        ParameterData{entiall}.prop2=0;
        ParameterData{entiall}.prop3=0;
        ParameterData{entiall}.prop4=0;

        ParameterData{entiall}.t=[0 0 0 1 1 2 2 3 3 3];

        ParameterData{entiall}.w=[1 wodd 1 wodd 1 wodd 1];

        ParameterData{entiall}.p=PP;

        ParameterData{entiall}.v=[0 3];

        ParameterData{entiall}.xnorm=0;
        ParameterData{entiall}.ynorm=0;
        ParameterData{entiall}.znorm=1;

        % NURBS curve

        ParameterData{entiall}.nurbs.form='B-NURBS';

        ParameterData{entiall}.nurbs.dim=4;

        ParameterData{entiall}.nurbs.number=7;

        ParameterData{entiall}.nurbs.coefs=zeros(4,7);
        ParameterData{entiall}.nurbs.coefs(4,:)=[1 wodd 1 wodd 1 wodd 1];

        ParameterData{entiall}.nurbs.coefs(1:3,:)=PP;

        ParameterData{entiall}.nurbs.coefs(1:3,:)=repmat(ParameterData{entiall}.nurbs.coefs(4,:),3,1).*ParameterData{entiall}.nurbs.coefs(1:3,:);

        ParameterData{entiall}.nurbs.order=3;

        ParameterData{entiall}.nurbs.knots=[0 0 0 1 1 2 2 3 3 3];

        [ParameterData{entiall}.dnurbs,ParameterData{entiall}.d2nurbs]=nrbDerivativesIGES(ParameterData{entiall}.nurbs);

        nup=500;
        p = nrbevalIGES(ParameterData{entiall}.nurbs,linspace(ParameterData{entiall}.v(1),ParameterData{entiall}.v(2),nup));
        len=sum(sqrt(sum((p(:,1:(nup-1))-p(:,2:nup)).^2,1)));
        if norm(p(:,1)-p(:,nup))<1e-3
            ParameterData{entiall}.length=3*len;
        else
            ParameterData{entiall}.length=min((len/norm(p(:,1)-p(:,nup))-1)*10+1,3)*len;
        end

        ParameterData{entiall}.well=true;

        clear PP nup p len

    elseif type==104

        A=Pvec(2);
        B=Pvec(3);
        C=Pvec(4);
        D=Pvec(5);
        E=Pvec(6);
        F=Pvec(7);
        ZT=Pvec(8);
        X1=Pvec(9);
        Y1=Pvec(10);
        X2=Pvec(11);
        Y2=Pvec(12);

        Q=[A 0.5*B;0.5*B C];

        [P,Dig] = eig(Q);

        if and(Dig(1,1)>1e-6,Dig(2,2)>1e-6)

            iDig=[1/Dig(1,1) 0;0 1/Dig(2,2)];

            R2=0.25*[D E]*P*iDig*P'*[D;E]-F;

            sqD11=sqrt(Dig(1,1));
            sqD22=sqrt(Dig(2,2));

            t=0.5*iDig*P'*[D;E];

            if det(P)>0
                pst=P'*[X1;Y1]+t;
                pen=P'*[X2;Y2]+t;
            else
                pen=P'*[X1;Y1]+t;
                pst=P'*[X2;Y2]+t;
            end

            pst(1)=pst(1)*sqD11;
            pst(2)=pst(2)*sqD22;

            pen(1)=pen(1)*sqD11;
            pen(2)=pen(2)*sqD22;

            if det([pst pen])<=0
                beta=2*pi-acos(dot(pst,pen)/R2);
            else
                beta=acos(dot(pst,pen)/R2);
            end

            if beta<1e-12
                beta=2*pi;
            end

            wodd=cos(beta/6);

            P0=pst;

            P1=[cos(beta/6) -sin(beta/6);sin(beta/6) cos(beta/6)]*pst/wodd;

            P2=[cos(beta/3) -sin(beta/3);sin(beta/3) cos(beta/3)]*pst;

            P3=[cos(beta/2) -sin(beta/2);sin(beta/2) cos(beta/2)]*pst/wodd;

            P4=[cos(2*beta/3) -sin(2*beta/3);sin(2*beta/3) cos(2*beta/3)]*pst;

            P5=[cos(5*beta/6) -sin(5*beta/6);sin(5*beta/6) cos(5*beta/6)]*pst/wodd;

            P6=pen;

            PP=[P0 P1 P2 P3 P4 P5 P6;ZT*ones(1,7)];

            PP(1,:)=PP(1,:)/sqD11-t(1);
            PP(2,:)=PP(2,:)/sqD22-t(2);
            PP(1:2,:)=P*PP(1:2,:);

            if useTrnsfrmEntity
                PP=rotMat*PP+repmat(transVec,1,7);
                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end

            ParameterData{entiall}.name='B-NURBS CRV';
            ParameterData{entiall}.original=0;
            ParameterData{entiall}.previous_type=104;
            ParameterData{entiall}.previous_name='CONIC ARC';

            ParameterData{entiall}.k=6;
            ParameterData{entiall}.m=2;

            ParameterData{entiall}.prop1=1;
            ParameterData{entiall}.prop2=0;
            ParameterData{entiall}.prop3=0;
            ParameterData{entiall}.prop4=0;

            ParameterData{entiall}.t=[0 0 0 1 1 2 2 3 3 3];

            ParameterData{entiall}.w=[1 wodd 1 wodd 1 wodd 1];

            ParameterData{entiall}.p=PP;

            ParameterData{entiall}.v=[0 3];

            ParameterData{entiall}.xnorm=0;
            ParameterData{entiall}.ynorm=0;
            ParameterData{entiall}.znorm=1;

            % NURBS curve

            ParameterData{entiall}.nurbs.form='B-NURBS';

            ParameterData{entiall}.nurbs.dim=4;

            ParameterData{entiall}.nurbs.number=7;

            ParameterData{entiall}.nurbs.coefs=zeros(4,7);
            ParameterData{entiall}.nurbs.coefs(4,:)=[1 wodd 1 wodd 1 wodd 1];

            ParameterData{entiall}.nurbs.coefs(1:3,:)=PP;

            ParameterData{entiall}.nurbs.coefs(1:3,:)=repmat(ParameterData{entiall}.nurbs.coefs(4,:),3,1).*ParameterData{entiall}.nurbs.coefs(1:3,:);

            ParameterData{entiall}.nurbs.order=3;

            ParameterData{entiall}.nurbs.knots=[0 0 0 1 1 2 2 3 3 3];

            [ParameterData{entiall}.dnurbs,ParameterData{entiall}.d2nurbs]=nrbDerivativesIGES(ParameterData{entiall}.nurbs);

            nup=500;
            p = nrbevalIGES(ParameterData{entiall}.nurbs,linspace(ParameterData{entiall}.v(1),ParameterData{entiall}.v(2),nup));
            len=sum(sqrt(sum((p(:,1:(nup-1))-p(:,2:nup)).^2,1)));
            if norm(p(:,1)-p(:,nup))<1e-3
                ParameterData{entiall}.length=3*len;
            else
                ParameterData{entiall}.length=min((len/norm(p(:,1)-p(:,nup))-1)*10+1,3)*len;
            end

            ParameterData{entiall}.well=true;

            clear PP nup p len iDig

        else

            p1=[X1;Y1;ZT];
            p2=[X2;Y2;ZT];

            if useTrnsfrmEntity
                p1=rotMat*p1+transVec;
                p2=rotMat*p2+transVec;
                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end

            ParameterData{entiall}.type=110;
            entty(110)=entty(110)+1;

            ParameterData{entiall}.name='LINE';
            ParameterData{entiall}.original=0;
            ParameterData{entiall}.previous_type=104;
            ParameterData{entiall}.previous_name='CONIC ARC';

            ParameterData{entiall}.p1=p1;
            ParameterData{entiall}.x1=p1(1);
            ParameterData{entiall}.y1=p1(2);
            ParameterData{entiall}.z1=p1(3);

            ParameterData{entiall}.p2=p2;
            ParameterData{entiall}.x2=p2(1);
            ParameterData{entiall}.y2=p2(2);
            ParameterData{entiall}.z2=p2(3);

            ParameterData{entiall}.length=norm(p1-p2);

            ParameterData{entiall}.well=false;

        end

        ParameterData{entiall}.a=A;
        ParameterData{entiall}.b=B;
        ParameterData{entiall}.c=C;
        ParameterData{entiall}.d=D;
        ParameterData{entiall}.e=E;
        ParameterData{entiall}.f=F;
        ParameterData{entiall}.zt=ZT;
        ParameterData{entiall}.x1=X1;
        ParameterData{entiall}.y1=Y1;
        ParameterData{entiall}.x2=X2;
        ParameterData{entiall}.y2=Y2;

        if showlines
            ParameterData{entiall}.superior=0;
        else
            ParameterData{entiall}.superior=1;
        end

        clear A B C D E F ZT X1 X2 Y1 Y2 P Dig Q

    elseif type==110

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='LINE';
        ParameterData{entiall}.original=1;

        if showlines
            ParameterData{entiall}.superior=0;
        else
            ParameterData{entiall}.superior=1;
        end

        ParameterData{entiall}.form=str2num(char(c((i*(80+nwro)+65):(i*(80+nwro)+72))));

        p1=Pvec(2:4)';
        p2=Pvec(5:7)';

        if useTransformationEnityArb
            if useTrnsfrmEntity
                p1=rotMat*p1+transVec;
                p2=rotMat*p2+transVec;
                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end
        end

        ParameterData{entiall}.p1=p1;
        ParameterData{entiall}.x1=p1(1);
        ParameterData{entiall}.y1=p1(2);
        ParameterData{entiall}.z1=p1(3);

        ParameterData{entiall}.p2=p2;
        ParameterData{entiall}.x2=p2(1);
        ParameterData{entiall}.y2=p2(2);
        ParameterData{entiall}.z2=p2(3);

        ParameterData{entiall}.length=norm(p1-p2);

        ParameterData{entiall}.well=true;

        clear p1 p2

    elseif type==102

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='COMPOSITE CRV';

        ParameterData{entiall}.n=Pvec(2);
        ParameterData{entiall}.de=round((Pvec(3:(2+Pvec(2)))+1)/2);

        ParameterData{entiall}.lengthcnt=zeros(1,Pvec(2));
        ParameterData{entiall}.length=0;

        ParameterData{entiall}.well=true;

    elseif type==142

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='CRV ON A PARAMETRIC SURFACE';

        ParameterData{entiall}.crtn=Pvec(2);
        ParameterData{entiall}.sptr=round((Pvec(3)+1)/2);
        ParameterData{entiall}.bptr=round((Pvec(4)+1)/2);
        ParameterData{entiall}.cptr=round((Pvec(5)+1)/2);
        ParameterData{entiall}.pref=Pvec(6);
        ParameterData{entiall}.length=0;

        ParameterData{entiall}.well=true;

        % POINT

    elseif type==116

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='POINT';

        ParameterData{entiall}.p=Pvec(2:4)';

        if useTransformationEnityArb
            if useTrnsfrmEntity
                ParameterData{entiall}.p=rotMat*ParameterData{entiall}.p+transVec;
                useTrnsfrmEntity=false;
                rotMat=eye(3);
                transVec=zeros(3,1);
            end
        end

        ParameterData{entiall}.x=ParameterData{entiall}.p(1);
        ParameterData{entiall}.y=ParameterData{entiall}.p(2);
        ParameterData{entiall}.z=ParameterData{entiall}.p(3);

        try
            ParameterData{entiall}.ptr=round((Pvec(5)+1)/2);
        catch
            ParameterData{entiall}.ptr=0;
        end

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.well=true;

        % OTHER

    elseif type==124

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='TRANSFORMATION MATRIX';

        ParameterData{entiall}.R=zeros(3);
        ParameterData{entiall}.T=zeros(3,1);

        ParameterData{entiall}.R(1,1)=Pvec(2);
        ParameterData{entiall}.R(1,2)=Pvec(3);
        ParameterData{entiall}.R(1,3)=Pvec(4);

        ParameterData{entiall}.T(1)=Pvec(5);

        ParameterData{entiall}.R(2,1)=Pvec(6);
        ParameterData{entiall}.R(2,2)=Pvec(7);
        ParameterData{entiall}.R(2,3)=Pvec(8);

        ParameterData{entiall}.T(2)=Pvec(9);

        ParameterData{entiall}.R(3,1)=Pvec(10);
        ParameterData{entiall}.R(3,2)=Pvec(11);
        ParameterData{entiall}.R(3,3)=Pvec(12);

        ParameterData{entiall}.T(3)=Pvec(13);

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.well=true;

        useTrnsfrmEntity=true;
        transVec=rotMat*(ParameterData{entiall}.T)+transVec;
        rotMat=rotMat*(ParameterData{entiall}.R);

        %         transVec=(ParameterData{entiall}.R)*transVec+ParameterData{entiall}.T;
        %         rotMat=(ParameterData{entiall}.R)*rotMat;

    elseif type==314

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='COLOR';

        inn=find(or(Pstr==44,Pstr==59));

        ParameterData{entiall}.cc1=str2num(char(Pstr((inn(1)+1):(inn(2)-1))));
        ParameterData{entiall}.cc2=str2num(char(Pstr((inn(2)+1):(inn(3)-1))));
        ParameterData{entiall}.cc3=str2num(char(Pstr((inn(3)+1):(inn(4)-1))));

        if length(inn)>4
            inn2=find(Pstr(1:(inn(5)-1))==72);
            if isempty(inn2)
                ParameterData{entiall}.cname='';
            else
                ParameterData{entiall}.cname=char(Pstr((inn2(1)+1):(inn(5)-1)));
            end
        else
            ParameterData{entiall}.cname='';
        end

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.well=true;

    elseif type==404

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='TYPE 404';
        ParameterData{entiall}.unknown=char(Pstr);

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.well=false;

    elseif type==406

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='TYPE 406';
        ParameterData{entiall}.unknown=char(Pstr);

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.well=false;

    elseif type==410

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='TYPE 410';
        ParameterData{entiall}.unknown=char(Pstr);

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.well=false;

    else

        entty(type)=entty(type)+1;

        ParameterData{entiall}.name='Unknown type!';
        entunk(type)=entunk(type)+1;

        ParameterData{entiall}.original=1;

        ParameterData{entiall}.length=0.00000000001;

        ParameterData{entiall}.well=false;

    end

end

num144=entty(144);

ent_ind=1:520;
EntityType=ent_ind(entty>0);
numEntityType=entty(entty>0);
unknownEntityType=ent_ind(entunk>0);
numunknownEntityType=entunk(entunk>0);

if not(isempty(unknownEntityType))
    disp(' ');
    disp(['Warning: There are unknown entity types for iges2matlab in ',igsfile,'.']);
    disp(' ');
    disp('Use "I-DEAS 3D IGES Translator" with NURBS as surface representation instead.');
    disp('If you dont have that posibility you can add IGES entities into iges2matlab().');
    disp('The IGES manual is found at');
    disp('www.uspro.org/documents/IGES5-3_forDownload.pdf');
    disp(' ');
end

cp1min=Inf;
cp1max=-Inf;
cp2min=Inf;
cp2max=-Inf;
cp3min=Inf;
cp3max=-Inf;

for i=1:noent

    if ParameterData{i}.type==144

        entiall=ParameterData{i}.pts;

        ParameterData{entiall}.superior=1;

        if ParameterData{i}.n1
            ParameterData=unDoSupMkC(ParameterData,ParameterData{i}.pto);
        end

        for j=1:ParameterData{i}.n2
            ParameterData=unDoSupMkC(ParameterData,ParameterData{i}.pti(j));
        end

        ParameterData{i}.nument=num144;

        if not(ParameterData{entiall}.original)
            if and(ParameterData{entiall}.previous_type==120,ParameterData{entiall}.well)

                ParameterData{i}.original=0;

                ptrRotCrv=ParameterData{entiall}.c;

                if ParameterData{ptrRotCrv}.type==110

                    CRVk=1;
                    CRVm=1;
                    CRVt=[0 0 1 1];
                    CRVw=[1 1];
                    CRVp=[ParameterData{ptrRotCrv}.p1 ParameterData{ptrRotCrv}.p2];
                    CRVv=[0 1];

                elseif ParameterData{ptrRotCrv}.type==126

                    CRVk=ParameterData{ptrRotCrv}.k;
                    CRVm=ParameterData{ptrRotCrv}.m;
                    CRVt=ParameterData{ptrRotCrv}.t;
                    CRVw=ParameterData{ptrRotCrv}.w;
                    CRVp=ParameterData{ptrRotCrv}.p;
                    CRVv=ParameterData{ptrRotCrv}.v;

                end

                p1=ParameterData{ParameterData{entiall}.l}.p1;

                rotDir=ParameterData{ParameterData{entiall}.l}.p2-p1;
                cpDirs=CRVp;
                cpDirs(1,:)=cpDirs(1,:)-p1(1);
                cpDirs(2,:)=cpDirs(2,:)-p1(2);
                cpDirs(3,:)=cpDirs(3,:)-p1(3);
                cpDirs=[0 -rotDir(3) rotDir(2);rotDir(3) 0 -rotDir(1);-rotDir(2) rotDir(1) 0]*cpDirs;
                lDist=sum(cpDirs.^2,1);
                [sqDist,maInd]=max(lDist);

                A=[p1 ParameterData{ParameterData{entiall}.l}.p2 p1+cpDirs(:,maInd)];
                B=zeros(3,3);
                B(3,2)=norm(rotDir);
                B(2,3)=sqrt(sqDist);

                meA=mean(A,2);
                meB=mean(B,2);

                [U,Sigm,V]=svd((B-repmat(meB,1,3))*(A-repmat(meA,1,3))');
                R=U*V';
                detR=det(R);
                if detR<0
                    V(:,3)=-V(:,3);
                    R=U*V';
                end
                T=meB-R*meA;

                CRVp=R*CRVp+repmat(T,1,CRVk+1);

                vmima=[7,-1];

                oCptr=ParameterData{ParameterData{i}.pto}.cptr;

                anglInInterval=-1;

                maxAbsz=max([CRVp(3,1),CRVp(3,end)]);
                minAbsz=min([CRVp(3,1),CRVp(3,end)]);

                meanz=mean([CRVp(3,1),CRVp(3,end)]);
                zmima=[maxAbsz,minAbsz];

                isUbounded=false;

                if ParameterData{oCptr}.type==102

                    for jj=1:ParameterData{oCptr}.n

                        PouterCrv=retSrfCrvPnt(2,ParameterData,1,ParameterData{oCptr}.de(jj),101,3);
                        PouterCrv=R*PouterCrv+repmat(T,1,101);

                        outerDist=PouterCrv(1,:).^2+PouterCrv(2,:).^2;

                        maxOuterDist=max(outerDist);
                        useOuter=outerDist>0.1*maxOuterDist;

                        angl=atan2(PouterCrv(2,useOuter),PouterCrv(1,useOuter));

                        for ii=1:length(angl)
                            if abs(angl(ii))<1e-4
                                angl(ii)=0;
                            elseif abs(angl(ii))>3.1415
                                angl(ii)=pi;
                            elseif angl(ii)<0
                                angl(ii)=angl(ii)+2*pi;
                            end
                        end

                        if var(angl)<1e-10
                            outerDist=mean(angl);
                            vmima(1)=min(outerDist,vmima(1));
                            vmima(2)=max(outerDist,vmima(2));
                            meanz=mean(PouterCrv(3,:));
                        else
                            anglInInterval=mean(angl);

                            outerDist=mean(PouterCrv(3,:));
                            zmima(1)=min(outerDist,zmima(1));
                            zmima(2)=max(outerDist,zmima(2));

                            isUbounded=true;
                        end

                    end

                else

                    PouterCrv=retSrfCrvPnt(2,ParameterData,0,ParameterData{ParameterData{i}.pto}.cptr,101,3);
                    PouterCrv=R*PouterCrv+repmat(T,1,101);

                    outerDist=PouterCrv(1,:).^2+PouterCrv(2,:).^2;

                    maxOuterDist=max(outerDist);
                    useOuter=outerDist>0.1*maxOuterDist;

                    angl=atan2(PouterCrv(2,useOuter),PouterCrv(1,useOuter));

                    for ii=1:length(angl)
                        if abs(angl(ii))<1e-4
                            angl(ii)=0;
                        elseif abs(angl(ii))>3.1415
                            angl(ii)=pi;
                        elseif angl(ii)<0
                            angl(ii)=angl(ii)+2*pi;
                        end
                    end

                    vmima(1)=min(min(angl),vmima(1));
                    vmima(2)=max(max(angl),vmima(2));
                    anglInInterval=mean(angl);

                end

                if vmima(2)<0
                    vmima(1)=0;
                    vmima(2)=2*pi;
                end

                vdiff=vmima(2)-vmima(1);

                if anglInInterval<0
                    if and(vdiff>3.1416,vdiff<6.28)
                        vdiff=2*pi-vdiff;

                        vmima(1)=vmima(2);
                        vmima(2)=vmima(1)+vdiff;
                    end
                else
                    if not(and(anglInInterval>vmima(1),anglInInterval<vmima(2)))
                        vdiff=2*pi-vdiff;

                        vmima(1)=vmima(2);
                        vmima(2)=vmima(1)+vdiff;
                    end
                end

                wodd=cos(vdiff/6);

                PDw=[CRVw;wodd*(CRVw);CRVw;wodd*(CRVw);CRVw;wodd*(CRVw);CRVw]';
                PDp=zeros(3,CRVk+1,7);

                betavec=linspace(vmima(1),vmima(2),7);

                eve=true;

                for ii=1:7

                    if eve
                        cob=cos(betavec(ii));
                        sib=sin(betavec(ii));
                        eve=false;
                    else
                        cob=cos(betavec(ii))/wodd;
                        sib=sin(betavec(ii))/wodd;
                        eve=true;
                    end

                    PDp(1,:,ii)=cob*CRVp(1,:)-sib*CRVp(2,:)-T(1);
                    PDp(2,:,ii)=sib*CRVp(1,:)+cob*CRVp(2,:)-T(2);
                    PDp(3,:,ii)=CRVp(3,:)-T(3);
                    PDp(:,:,ii)=R'*PDp(:,:,ii);

                end

                ParameterData{i}.u=CRVv;

                if isUbounded
                    if and(zmima(1)>(minAbsz+1e-6),zmima(2)<(maxAbsz-1e-6))
                        zdiff=zmima(2)-zmima(1);
                        if ParameterData{ptrRotCrv}.type==126
                            if zdiff<1e-2

                                boundz=0.5*(zmima(1)+zmima(2));
                                tmin=nrbCrvPlaneIntrsctIGES(ParameterData{ptrRotCrv}.nurbs,ParameterData{ptrRotCrv}.dnurbs,ParameterData{ptrRotCrv}.d2nurbs,R(3,:),boundz-T(3));

                                if CRVp(3,1)>boundz
                                    if meanz>boundz
                                        ParameterData{i}.u=[ParameterData{ptrRotCrv}.v(1) tmin];
                                    else
                                        ParameterData{i}.u=[tmin ParameterData{ptrRotCrv}.v(2)];
                                    end
                                else
                                    if meanz<boundz
                                        ParameterData{i}.u=[ParameterData{ptrRotCrv}.v(1) tmin];
                                    else
                                        ParameterData{i}.u=[tmin ParameterData{ptrRotCrv}.v(2)];
                                    end
                                end

                            else

                                paramsvec=[0,0];
                                paramsvec(1)=nrbCrvPlaneIntrsctIGES(ParameterData{ptrRotCrv}.nurbs,ParameterData{ptrRotCrv}.dnurbs,ParameterData{ptrRotCrv}.d2nurbs,R(3,:),zmima(1)-T(3));
                                paramsvec(2)=nrbCrvPlaneIntrsctIGES(ParameterData{ptrRotCrv}.nurbs,ParameterData{ptrRotCrv}.dnurbs,ParameterData{ptrRotCrv}.d2nurbs,R(3,:),zmima(2)-T(3));

                                if paramsvec(2)>paramsvec(1)
                                    ParameterData{i}.u=paramsvec;
                                else
                                    ParameterData{i}.u=[paramsvec(2),paramsvec(1)];
                                end

                            end
                        elseif ParameterData{ptrRotCrv}.type==110

                            if zdiff<1e-2

                                boundz=0.5*(zmima(1)+zmima(2));

                                tmin=(boundz-CRVp(3,1))/(CRVp(3,2)-CRVp(3,1));

                                if tmin<0
                                    tmin=0;
                                elseif tmin>1
                                    tmin=1;
                                end

                                if CRVp(3,1)>boundz
                                    if meanz>boundz
                                        ParameterData{i}.u=[ParameterData{ptrRotCrv}.v(1) tmin];
                                    else
                                        ParameterData{i}.u=[tmin ParameterData{ptrRotCrv}.v(2)];
                                    end
                                else
                                    if meanz<boundz
                                        ParameterData{i}.u=[ParameterData{ptrRotCrv}.v(1) tmin];
                                    else
                                        ParameterData{i}.u=[tmin ParameterData{ptrRotCrv}.v(2)];
                                    end
                                end

                            else

                                paramsvec=[0,0];

                                for ii=1:2
                                    paramsvec(ii)=(zmima(ii)-CRVp(3,1))/(CRVp(3,2)-CRVp(3,1));
                                    if paramsvec(ii)<0
                                        paramsvec(ii)=0;
                                    elseif paramsvec(ii)>1
                                        paramsvec(ii)=1;
                                    end
                                end

                                if paramsvec(2)>paramsvec(1)
                                    ParameterData{i}.u=paramsvec;
                                else
                                    ParameterData{i}.u=[paramsvec(2),paramsvec(1)];
                                end

                            end
                        end
                    end
                end

                ParameterData{i}.v=[0 vdiff];

                % NURBS surface

                ParameterData{i}.nurbs.form='B-NURBS';
                ParameterData{i}.nurbs.dim=4;
                ParameterData{i}.nurbs.number=[CRVk+1 7];
                ParameterData{i}.nurbs.coefs=zeros(4,CRVk+1,7);

                for ii=1:7
                    ParameterData{i}.nurbs.coefs(4,:,ii)=PDw(:,ii)';

                    ParameterData{i}.nurbs.coefs(1,:,ii)=ParameterData{i}.nurbs.coefs(4,:,ii).*PDp(1,:,ii);
                    ParameterData{i}.nurbs.coefs(2,:,ii)=ParameterData{i}.nurbs.coefs(4,:,ii).*PDp(2,:,ii);
                    ParameterData{i}.nurbs.coefs(3,:,ii)=ParameterData{i}.nurbs.coefs(4,:,ii).*PDp(3,:,ii);
                end

                ParameterData{i}.nurbs.knots=cell(1,2);
                ParameterData{i}.nurbs.knots{1}=CRVt;
                ParameterData{i}.nurbs.knots{2}=[0 0 0 1/3 1/3 2/3 2/3 1 1 1]*vdiff;

                ParameterData{i}.nurbs.order=[CRVm+1 3];

                [ParameterData{i}.dnurbs,ParameterData{i}.d2nurbs]=nrbDerivativesIGES(ParameterData{i}.nurbs);

            end
        end

    elseif ParameterData{i}.type==102

        for j=1:ParameterData{i}.n
            ParameterData{i}.lengthcnt(j)=ParameterData{ParameterData{i}.de(j)}.length;
        end

        ParameterData{i}.length=sum(ParameterData{i}.lengthcnt);

    elseif ParameterData{i}.type==128

        cp1min=min(cp1min,min(reshape(ParameterData{i}.p(1,:,:),1,[])));
        cp1max=max(cp1max,max(reshape(ParameterData{i}.p(1,:,:),1,[])));
        cp2min=min(cp2min,min(reshape(ParameterData{i}.p(2,:,:),1,[])));
        cp2max=max(cp2max,max(reshape(ParameterData{i}.p(2,:,:),1,[])));
        cp3min=min(cp3min,min(reshape(ParameterData{i}.p(3,:,:),1,[])));
        cp3max=max(cp3max,max(reshape(ParameterData{i}.p(3,:,:),1,[])));

    end
end

gdiag=norm([cp1max-cp1min,cp2max-cp2min,cp3max-cp3min]);

for i=1:noent
    if ParameterData{i}.type==142
        ParameterData{i}.length=ParameterData{ParameterData{i}.cptr}.length;
        ParameterData{i}.gdiagonal=gdiag;
    elseif ParameterData{i}.type==144
        ParameterData{i}.gdiagonal=gdiag;
    end
end

% Recursive define function

function ParameterData=unDoSupMkC(ParameterData,ii)

if ParameterData{ii}.type==126
    ParameterData{ii}.superior=0;
elseif ParameterData{ii}.type==110
    ParameterData{ii}.superior=0;
elseif ParameterData{ii}.type==102
    for k=1:ParameterData{ii}.n
        ParameterData=unDoSupMkC(ParameterData,ParameterData{ii}.de(k));
    end
elseif ParameterData{ii}.type==142
    % only cptr, not bptr
    ParameterData=unDoSupMkC(ParameterData,ParameterData{ii}.cptr);

    ParameterData=makeContinous(ParameterData,ParameterData{ii}.bptr);
    ParameterData=makeContinous(ParameterData,ParameterData{ii}.cptr);
end


function ParameterData=makeContinous(ParameterData,ii)

allLines=false;

if ParameterData{ii}.type==102
    if ParameterData{ii}.n>1
        [stPp,endPp,isLine,isKnown]=endPoints(ParameterData,ParameterData{ii}.de(1));
        if isKnown
            allLines=isLine;
            [stP,endP,isLine,isKnown]=endPoints(ParameterData,ParameterData{ii}.de(2));
            if isKnown
                if allLines
                    allLines=isLine;
                end

                if ((endPp(1)-stP(1))^2+(endPp(2)-stP(2))^2+(endPp(3)-stP(3))^2)>1e-8
                    if ((endPp(1)-endP(1))^2+(endPp(2)-endP(2))^2+(endPp(3)-endP(3))^2)<1e-8
                        ParameterData=mkreverse(ParameterData,ParameterData{ii}.de(2));
                        endPp=stP;
                    elseif ((stPp(1)-stP(1))^2+(stPp(2)-stP(2))^2+(stPp(3)-stP(3))^2)<1e-8
                        ParameterData=mkreverse(ParameterData,ParameterData{ii}.de(1));
                        endPp=endP;
                    elseif ((stPp(1)-endP(1))^2+(stPp(2)-endP(2))^2+(stPp(3)-endP(3))^2)<1e-8
                        ParameterData=mkreverse(ParameterData,ParameterData{ii}.de(1));
                        ParameterData=mkreverse(ParameterData,ParameterData{ii}.de(2));
                        endPp=stP;
                    else
                        endPp=endP;
                    end
                else
                    endPp=endP;
                end

                for k=3:ParameterData{ii}.n
                    [stP,endP,isLine,isKnown]=endPoints(ParameterData,ParameterData{ii}.de(k));
                    if allLines
                        allLines=isLine;
                    end
                    if isKnown
                        if ((endPp(1)-endP(1))^2+(endPp(2)-endP(2))^2+(endPp(3)-endP(3))^2)<1e-8
                            ParameterData=mkreverse(ParameterData,ParameterData{ii}.de(k));
                            endPp=stP;
                        else
                            endPp=endP;
                        end
                    else
                        break
                    end
                end

            end
        end
    end
end

ParameterData{ii}.allLines=allLines;

function [stP,endP,isLine,isKnown]=endPoints(ParameterData,ii)

if ParameterData{ii}.type==126
    stP=ParameterData{ii}.p(:,1);
    endP=ParameterData{ii}.p(:,end);
    isLine=false;
    isKnown=true;
elseif ParameterData{ii}.type==110
    stP=ParameterData{ii}.p1;
    endP=ParameterData{ii}.p2;
    isLine=true;
    isKnown=true;
else
    stP=[0;0;0];
    endP=[0;0;0];
    isLine=false;
    isKnown=false;
end

function ParameterData=mkreverse(ParameterData,ii)

if ParameterData{ii}.type==110

    p1=[ParameterData{ii}.x1;ParameterData{ii}.y1;ParameterData{ii}.z1];
    p2=[ParameterData{ii}.x2;ParameterData{ii}.y2;ParameterData{ii}.z2];

    ParameterData{ii}.p1=p2;
    ParameterData{ii}.p2=p1;

    ParameterData{ii}.x1=p2(1);
    ParameterData{ii}.y1=p2(2);
    ParameterData{ii}.z1=p2(3);

    ParameterData{ii}.x2=p1(1);
    ParameterData{ii}.y2=p1(2);
    ParameterData{ii}.z2=p1(3);

elseif ParameterData{ii}.type==126

    tsum=ParameterData{ii}.t(1)+ParameterData{ii}.t(end);

    t=tsum-ParameterData{ii}.t;

    ParameterData{ii}.t=t(end:(-1):1);
    ParameterData{ii}.w=ParameterData{ii}.w(end:(-1):1);
    ParameterData{ii}.p=ParameterData{ii}.p(:,end:(-1):1);

    ParameterData{ii}.v=[tsum-ParameterData{ii}.v(2) tsum-ParameterData{ii}.v(1)];

    ParameterData{ii}.nurbs.coefs=ParameterData{ii}.nurbs.coefs(:,end:(-1):1);

    ParameterData{ii}.nurbs.knots=ParameterData{ii}.t;

elseif ParameterData{ii}.type==102

    de=ParameterData{ii}.de;
    for k=1:ParameterData{ii}.n
        ParameterData{ii}.de(k)=de(ParameterData{ii}.n+1-k);
        ParameterData=mkreverse(ParameterData,ParameterData{ii}.de(k));
    end

elseif ParameterData{ii}.type==142

    ParameterData=mkreverse(ParameterData,ParameterData{ii}.bptr);
    ParameterData=mkreverse(ParameterData,ParameterData{ii}.cptr);

end
