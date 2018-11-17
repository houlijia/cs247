% demo_video_acquisition
% Demonstrate the coding method using compressive video sampling  
% for data acquisition model.
%
% Input       : .yuv video 
% Measurements: permuted Walsh Hadamard transform (real)
% 
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 06/14/2010

clear all;  close all;
path(path,genpath(pwd));
fullscreen = get(0,'ScreenSize');

% problem size
ratio = .1;
slx = 288;
sly = 352;
nfm = 5;
N = slx*sly*nfm;
M = round(ratio*N);
scl = 30;
ndb = 5;          % SNR of Gaussian noise

% original image
%[cellY, cellU, cellV] = yuv_import('coastguard_cif_300.yuv',[sly slx],nfm);
[cellY, cellU, cellV] = yuv_import('news_cif_300.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('foreman_cif_300.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('bridge-close_cif_2001.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('bus_cif_150.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('flower_cif_250.yuv',[sly slx],nfm);
Y = zeros(slx,sly,nfm);
for i = 1:nfm
    Y(:,:,i) = cellY{i};
end
Y2 = Y/scl;

% generate measurement matrix
p = randperm(N);
picks = p(1:M);
picks(1) = 1;
for ii = 2:M
    if picks(ii) == 1
        picks(ii) = p(M+1);
        break;
    end
end
perm = randperm(N); % column permutations allowable
A = @(x,mode) pwht_dfA(x,picks,perm,mode);
%A = @(x,mode) pdct_dfA(x,picks,perm,mode);

% observation
b = A_fWH(Y2(:),picks,perm);

% add noise
noise = randn(N,1);          % length depends on bandwith
sigma = norm(b(2:end))/norm(noise)/abs(10^(ndb/20));     % neglect the first coeff
b(2:end) = b(2:end) + sigma*noise(2:M);

% reconstruction using TVAL3_CVS
clear opts
opts.mu = 2^6;               % correspond to ||Au-b||^2
opts.beta = 2^2;             % correspond to ||Du-w||^2
opts.mu0 = 4;
opts.beta0 = 1/4;
opts.tol = 1e-4;
opts.tol_inn = 1e-3;
opts.maxin = 12;
opts.maxcnt = 12;
opts.StpCr = 0;              % 0 (optimality gap) or 1 (relative change)
opts.disp = 10;
opts.Ut = Y2;
opts.aqst = true;

start_t = tic;
%profile on;
[estY2, out] = TVAL3_CVS_D2(A,b,slx,sly,nfm,opts);
%profile viewer;
estY = estY2*scl;
t = toc(start_t);
[MSE,PSNR]=iq_measures(Y,estY);

% % generate and play the video
% for i = 1:nfm
%     cell_estY{i} = estY(:,:,i);
% end
% delete('test.yuv');
% yuv_export(cell_estY, cellU, cellV, 'test.yuv', nfm);
% clear org;
% org = yuv2mov('test.yuv',sly,slx,'420');
% mplay(org,3);

% plotting
figure('Name','Comparison','Position',...
    [fullscreen(1) fullscreen(2)-100 fullscreen(3) fullscreen(4)-50]);
colormap(gray);

subplot(2,2,1);
imshow(Y(:,:,2),[0 255]);
title('Original frame 2','fontsize',16);

subplot(2,2,3);
imshow(estY(:,:,2),[0 255]);
title('Recovered frame 2','fontsize',16);
xlabel(sprintf('PSNR: %4.2f,   Noise: %2d%%',PSNR(2),100*sigma),'fontsize',14);

subplot(2,2,2);
imshow(abs(Y(:,:,end)), [0 255]);
title('Last frame','fontsize',16);

subplot(2,2,4);
title('Recovered last frame','fontsize',16);
imshow(abs(estY(:,:,end)), [0 255]);
xlabel(sprintf('PSNR: %4.2f',PSNR(end)),'fontsize',14);

figure('Name','Parameters','Position',...
    [fullscreen(1) fullscreen(2)-100 fullscreen(3) fullscreen(4)-50]);
semilogy(1:length(out.lam1),out.lam1,'b*:',1:length(out.lam2),sqrt(out.lam2),'rx:',...
    1:length(out.lam3),sqrt(out.lam3),'g.--', 1:length(out.f),sqrt(out.f),'m+-');
legend('lam1(||w||_1)','lam2(||D(d_tu)-w||_2)','lam3(||Au-b||_2)','obj function')

figure('Name','1D Comparison','Position',...
    [fullscreen(1) fullscreen(2)-100 fullscreen(3) fullscreen(4)-50]);

subplot(1,2,1);
plot(Y(1:10:end),'.');
title('Y');

subplot(1,2,2);
plot(estY(1:10:end),'rx');
title('Estimate Y');
