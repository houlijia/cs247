% demo_video
%
% demo the coding method using compressive video sampling 
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

file_root_of_compsens_repository='/vol0/yenminglai/compsens/';
path(path,[file_root_of_compsens_repository 'test_data/'])
path(path,[file_root_of_compsens_repository 'trunk/Fast_Walsh_Hadamard_Transform'])
path(path,[file_root_of_compsens_repository 'trunk/Quality Assessment Toolbox'])
fullscreen = get(0,'ScreenSize');

% problem size
ratio = .1;
slx = 288;
sly = 352;
nfm = 5;
N = slx*sly*nfm;
M = round(ratio*N);
scl = 1000;
ndb = inf;          % SNR of Gaussian noise

% original image
%[cellY, cellU, cellV] = yuv_import('coastguard_cif_300.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('news_cif_300.yuv',[sly slx],nfm);
[cellY, cellU, cellV] = yuv_import('foreman_cif_300.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('bridge-close_cif_2001.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('bus_cif_150.yuv',[sly slx],nfm);
%[cellY, cellU, cellV] = yuv_import('flower_cif_250.yuv',[sly slx],nfm);
Y = zeros(slx,sly,nfm);
for i = 1:nfm
    Y(:,:,i) = cellY{i};
end

% % truncate the temporal differences
% TY = T(Y);
% idx = find(abs(TY)>2);
% map = zeros(slx,sly,nfm);
% map(idx) = 1;
% TY2 = TY.*map;
% Y2 = Tinv(TY2);
% [MSE,PSNR]=iq_measures(Y2,Y);
% fprintf('Median PSNR after truncation: %4.2f. \n', median(PSNR));

dctY = reshape(temporal_dct(Y(:),slx,sly,nfm),slx,sly,nfm)/scl;
peak = max(max(max(abs(dctY))));
dctY2 = dctY;
for i = 2:nfm
    subpk(i) = max(max(abs(dctY(:,:,i))));
    idx = find(abs(dctY(:,:,i)) > subpk(i));
    dctY2((i-1)*slx*sly+idx) = dctY((i-1)*slx*sly+idx)*(peak/subpk(i));
end

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
perm = randperm(N);
A = @(x,mode) pwht_dfA(x,picks,perm,mode);
%A = @(x,mode) pdct_dfA(x,picks,perm,mode);

% observation
b = A(dctY2(:),1);

% add noise
noise = randn(N,1);          % length depends on bandwith
sigma = norm(b(2:end))/norm(noise(2:end))/abs(10^(ndb/20));
b = b + sigma*noise(1:M);
noisyY = Y + sigma*scl*randn(size(Y));

% reconstruction using TVAL3_CVS
clear opts
opts.mu = 2^11;               % correspond to ||Au-b||^2
opts.beta = 2^7;             % correspond to ||Du-w||^2
opts.mu0 = 2^7;
opts.beta0 = 2^3;
opts.tol = 1e-4;
opts.tol_inn = 1e-3;
opts.maxin = 12;
opts.maxcnt = 12;
opts.StpCr = 0;              % 0 (optimality gap) or 1 (relative change)
opts.disp = 10;
opts.Ut = dctY2;

start_t = tic;
% profile on;
%[estdctY2, out] = TVAL3_CVS_D2(A,b,slx,sly,nfm,opts);
[estdctY2, out] = TVAL3_CVS_D2(A,b,slx,sly,nfm,opts);
estdctY = estdctY2;
for i = 2:nfm
    idx = find(abs(estdctY(:,:,i)) > subpk(i));
    estdctY((i-1)*slx*sly+idx) = estdctY2((i-1)*slx*sly+idx)/(peak/subpk(i));
    %idx = find(abs(estdctY(:,:,i)) < .1*subpk(i));    %no critical differences
    %estdctY((i-1)*slx*sly+idx) = 0;
end
%estdctY = estdctY - mean(estdctY(:)) + midref;

estY = reshape(temporal_idct(estdctY(:),slx,sly,nfm),slx,sly,nfm)*scl;
% profile viewer;
t = toc(start_t);
[MSE,PSNR]=iq_measures(Y,estY);
[MSE2,PSNR2]=iq_measures(dctY,estdctY);

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

subplot(2,3,1);
imshow(Y(:,:,2),[0 255]);
title('Original frame 2','fontsize',16);

subplot(2,3,4);
imshow(estY(:,:,2),[0 255]);
title('Recovered frame 2','fontsize',16);
xlabel(sprintf('PSNR: %4.2f,   Noise: %3.1f dB',PSNR(2),ndb),'fontsize',14);

subplot(2,3,2);
imshow(abs(dctY(:,:,1)), [0 max(max(abs(dctY(:,:,1))))]);
title('First DCT frame','fontsize',16);

subplot(2,3,5);
imshow(abs(estdctY(:,:,1)), [0 max(max(abs(estdctY(:,:,1))))]);
title('Recovered first DCT frame','fontsize',16);
xlabel(sprintf('PSNR: %4.2f',PSNR2(1)),'fontsize',14);

subplot(2,3,3);
imshow(abs(dctY(:,:,end)), [0 max(max(abs(dctY(:,:,end))))]);
title('Last DCT frame','fontsize',16);

subplot(2,3,6);
imshow(abs(estdctY(:,:,end)), [0 max(max(abs(estdctY(:,:,end))))]);
xlabel(sprintf('PSNR: %4.2f',PSNR2(end)),'fontsize',14);

figure('Name','Parameters','Position',...
    [fullscreen(1) fullscreen(2)-100 fullscreen(3) fullscreen(4)-50]);
semilogy(1:length(out.lam1),out.lam1,'b*:',1:length(out.lam2),sqrt(out.lam2),'rx:',...
    1:length(out.lam3),sqrt(out.lam3),'g.--', 1:length(out.f),sqrt(out.f),'m+-');
legend('lam1(||w||_1)','lam2(||D(d_tu)-w||_2)','lam3(||Au-b||_2)','obj function')

figure('Name','1D Comparison','Position',...
    [fullscreen(1) fullscreen(2)-100 fullscreen(3) fullscreen(4)-50]);

subplot(2,2,1);
plot(dctY2(1:10:end),'.');
title('dctY');

subplot(2,2,2);
plot(estdctY2(1:10:end),'rx');
title('Estimate dctY');

subplot(2,2,3);
plot(Y(1:10:end),'.');
title('Y');

subplot(2,2,4);
plot(estY(1:10:end),'rx');
title('Estimate Y');
