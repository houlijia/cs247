% main_admm_image_cs_wavelet
% use the admm to solve the image cs problem while the sensing matrix is
% the permutation hadamard matrix and the wavelet transforamtion is used.
close all
clear all

clc

addpath(genpath('./funs'));

filename = 'Barbara.png';
I=imread(filename);

row = 256;
col = 256;

I=im2double(I);
I = imresize(I, [row,col]);
figure; subplot(2,2,1); imshow(I); axis off;
title('Ground truth');

load('perm64k.mat');
% P = 1:2^16;
P = P64k;
[B,Q] = sort(P);

CSr = 0.1;  
Mea_num = round(CSr*length(I(:)));


R = @(z) A_wht(z, Q, Mea_num);
RT = @(z) At_wht(z, P, Mea_num, row, col);

patch = 6;
step = 2;
T_dct = dctmtx(patch);
[row,  col]   =  size(I);


I2P = @(I) image2patches3d(I, patch, patch, step, step);
P2I = @(P) patches3d2image(P, row,col,step, step);

I2Bdct = @(I) dct_block2d(I, T_dct);
Bdct2I = @(P) dct_block2d(P, T_dct');


W = @(x)  myfastidct2_givenT(x,P2I,Bdct2I);
WT = @(x) myfastdct2_givenT(x,row,col,I2P,I2Bdct);

A = @(x) R(W(x));
AT = @(x) WT(RT(x));

% generate measurement
y = R(I(:));




%% IST SLOPE
para.row =row;
para.col = col;
para.CSr = CSr;
para.lambda = 2;
para.iter = 100;
para.patch = patch;
para.step = step;
para.ori_im = I;

f    =   IST_SLOPE( y, AT,para,I,W,WT,RT,R);


Im_rec_myIST = reshape(W(f),[row, col]);
psnr_rec_myIST = psnr(Im_rec_myIST, I);
subplot(2,2,2); 
%figure; 
imshow(Im_rec_myIST./max(Im_rec_myIST(:))); axis off;
title(['Rec by IST-SLOPE, PSNR: ' num2str(psnr_rec_myIST)]);

%% GAP_SLOPE


f    =   GAP_SLOPE( y, AT,para,I,W,WT,RT,R);

Im_rec_GAP = reshape(W(f),[row, col]);
psnr_rec_GAP = psnr(Im_rec_GAP, I);
subplot(2,2,3); 
%figure; 
imshow(Im_rec_GAP./max(Im_rec_GAP(:))); axis off;
title(['Rec by GAP-SLOPE, PSNR: ' num2str(psnr_rec_GAP)]);


%% start for admm
para.iter = 200;
para.beta = 1e-2;
para.eta = 1e-1;
Im_rec   =   ADMM_SLOPE( y,AT,para,I,W,WT,RT,R);

psnr_rec = psnr(Im_rec, I);
subplot(2,2,4); 
%figure; 
imshow(Im_rec./max(Im_rec(:)));
title(['Rec by ADMM-SLOPE, PSNR: ' num2str(psnr_rec)]);


para.beta0 = 1e-3;
para.eta0= 1e-3;
para.beta_max =1e-1;
para.eta_max=1e-1;
Im_rec   =   ADMM_SLOPE_en( y,AT,para,I,W,WT,RT,R);

psnr_rec = psnr(Im_rec, I);
subplot(2,2,4); 
%figure; 
imshow(Im_rec./max(Im_rec(:)));
title(['Rec by ADMM-SLOPE-en, PSNR: ' num2str(psnr_rec)]);


%% TV solution
para.lambda = 2;  
para.TVweight = 0.07;
nr = 1;
f_TV = TV_CSI( y, nr,para, R,RT);
figure; 
psnr_TV = psnr(f_TV, I);
subplot(1,2,1); 
%figure; 
imshow(f_TV./max(f_TV(:)));
title(['Rec by IST-TV, PSNR: ' num2str(psnr_TV)]);

para.lambda = 1;  
f_GAP_TV = TV_GAP( y, nr,para, R,RT);
%figure; 
psnr_GAP_TV = psnr(f_GAP_TV, I);
subplot(1,2,2); 
%figure; 
imshow(f_GAP_TV./max(f_GAP_TV(:)));
title(['Rec by GAP-TV, PSNR: ' num2str(psnr_GAP_TV)]);

