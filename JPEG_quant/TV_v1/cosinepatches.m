
%
% find mixing matix for a finite-size sensor
%
%
% Lo = length of pixel size in object plan
% Ls = length of sensor (sensor is a square)
% La = length of aperture
% Do = distance of object to sensor
% Da = distance of aperture to sensor
%
% Kx, Ky = factors for super resolution in x and y
% In each pixel of the oringal resolution, there is Kx pixles in x
% direction, and Ky pixels in y direction
%

close all

DaOverDo = 1/10;  %ratio of distance
LsOverLa = 1e-5;     %ratio of sensor size over apperture size
Nax=1;   % aperture elements in x
Nay=1;   % aperture elements in y
Px=8;      % patch size in terms of aperture elements x
Py=8;      % patch size in terms of aperture elements y
Pc=7;      % number of modes per patch, from 0 to Pc-1
Npx=floor(Nax/Px);
Npy=floor(Nay/Py);

Sx=LsOverLa/2;
Sy=Sx;

dt=1;
nb=2/dt;
t=(1:dt:(Px+nb*dt))'-1;

alpha=1/(1+DaOverDo) ;
%alpha= 0;

coef=zeros(Nax,Npx,Pc);

ims=length(t);
nbdr=0;
im=zeros(ims+2*nbdr,Pc);
zpad=1*ones(nbdr,1);

for ii=1:1
    i=ii-1;
    for pp=1:1
        p=pp-1;
        for mm=1:Pc
            m=mm-1;
%
% define patch which is zero outside of 
%      [p*Px , (p+1)*Px]x[q*Py , (q+1)*Py]
%
   fx = @(x,u) pcos(x-alpha*u,p,m,Px);
   c = integral2(fx,i,i+1,-Sx,Sx);
   coef(ii,pp,mm)=c;
   
   imx=fx(t,0*t);
   imx=imx/(max(imx));
   im(:,mm)=[zpad ; imx ; zpad] ; 
%
%  add some borders
%   
        end
    end
end

newimt=[];
newim=[];
for mm=1:Pc
    newimt=[];
    for kk=1:Pc
        newimt=[newimt im(:,mm)*im(:,kk)'];
    end
    newim=[newim ; newimt];
end

[tn,tm]=size(newim);

newim=[zeros(tn+nb/2,nb/2) [zeros(nb/2,tm) ; newim ] ]; 

figure ; imshow(newim,[-1 1], 'border','tight');