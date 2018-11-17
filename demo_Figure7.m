%main_test
% Just run this code you will see the results of Figure 7 in the main
% paper.
% The image 'boats.tif' is used, 2D-WHT, 2D-DCT, SRM-WHT, SRM-DCT is used for sensing matrix and
% NLR-CS is used for reconstruction.

% If you want to change images, please simply change the "filename" to one
% of the 8 ncandinates, The true images are stores in ./test_data/Image 

% The output CS files and reconstructed images are stored in
% ./test_data/Image and a folder with the filename

% Xin Yuan & Raziel Haimi-Cohen
% Nokia Bell Labs
% xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 

% For each image, the running time is about 30 minutes, this includes the
% encoding, decoding use NLR-CS for reconstruction.

% If you want to see a quick result, please run "demo_GAPTV_SRM.m"


clear all
close all
clc



cd ./tools/scripts
set_mex_path dbg-ng vc
cd ..
cd ..
addpath(genpath('../CS_JPEG_demo'));

cd ./JPEG_quant
addpath(genpath('../JPEG_quant'));
filename = 'boats';
img_files= {[filename '.tif']};
[stats, enc_opts] =  run_quant_test_cscd_test_2DSRM_WHTDCT_NLR_CS('../test_data/Image', '../test_data/Image',img_files);

cd ..
load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
h = figure; 
plot(sav_bytes,sav_ssim,'k-','LineWidth',1);
load(['.\test_data\Image\' filename '.jpeg2000_qual\JPEG2000_qual.mat'])
hold on
plot(sav_bytes,sav_ssim,'k--','LineWidth',1);

for ncs = 1:10
     file_size_cs_2DWHT(ncs) = stats((ncs-1)*4+1).n_byte;
     cs_ssim_NLR_2DWHT(ncs) = stats((ncs-1)*4+1).img_ssim;  
     
     file_size_cs_2DDCT(ncs) = stats((ncs-1)*4+2).n_byte;
     cs_ssim_NLR_2DDCT(ncs) = stats((ncs-1)*4+2).img_ssim; 
     
     file_size_cs_SRMWHT(ncs) = stats((ncs-1)*4+3).n_byte;
     cs_ssim_NLR_SRMWHT(ncs) = stats((ncs-1)*4+3).img_ssim;  
     
     file_size_cs_SRMDCT(ncs) = stats((ncs-1)*4+4).n_byte;
     cs_ssim_NLR_SRMDCT(ncs) = stats((ncs-1)*4+4).img_ssim;  
end
hold on;
plot(file_size_cs_2DWHT,cs_ssim_NLR_2DWHT,'r--o','LineWidth',1);
hold on
plot(file_size_cs_2DDCT,cs_ssim_NLR_2DDCT,'r-o','LineWidth',1);
hold on;
plot(file_size_cs_SRMWHT,cs_ssim_NLR_SRMWHT,'r--s','LineWidth',1);
hold on
plot(file_size_cs_SRMDCT,cs_ssim_NLR_SRMDCT,'r-s','LineWidth',1);

title(filename)
legend('JPEG', 'JPEG2000','CS (NLR-CS), 2D-WHT', 'CS (NLR-CS), 2D-DCT', 'CS (NLR-CS), SRM-WHT', 'CS (NLR-CS), SRM-DCT','Location','southeast');
set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
saveas(h, ['.\Temp_results\' filename '_JPEG_CS_NLR_CS_4SensingMatrix.fig']);
saveas(h, ['.\Temp_results\' filename '_JPEG_CS_NLR_CS_4SensingMatrix.png']);

save('.\Temp_results\demo_figure5.mat')


