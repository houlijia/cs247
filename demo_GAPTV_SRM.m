

% demo_GAPTV
% Just run this code you will see the results of Figure 4 in the main
% paper.
% All the 8 images are used. Both SRM-WHT and SRM-DCT are used for sensing matrix
% GAP-TV is used for reconstruction.
% therefore, it include 160 encoding times (8 images x 2 sensing matrices x 10 CSr)
% and 160 times reconstructions


% The output CS files and reconstructed images are stored in
% ./test_data/Image and a folder with the filename

% Xin Yuan & Raziel Haimi-Cohen
% Nokia Bell Labs
% xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 

% For each image, the running time is about 4 minutes

% If you want to see a quick result of one image, try "demo_firstTry.m"

% After your run this code to get the coded CS files, you can run

% "demo_read_CSfile_and_Reconstruction_SRM.m" to get results of other
% reconstruction algorithms.


clear all
close all
clc



cd ./tools/scripts
set_mex_path dbg-ng vc
cd ..
cd ..
addpath(genpath('../CS_JPEG_demo'));

%%
h8 = figure(100);
for nim = 1:8
    
cd ./JPEG_quant
addpath(genpath('../JPEG_quant'));
    switch  nim
        case 1
            filename = 'barbara';
        case 2
            filename = 'boats'; 
        case 3
            filename = 'cameraman';
        case 4
            filename = 'foreman';
        case 5
            filename = 'house';
        case 6
            filename = 'lena256';
        case 7
            filename = 'Monarch';
        case 8
            filename = 'Parrots';
    end
img_files= {[filename '.tif']};
[stats, enc_opts] =  run_quant_test_cscd_test_SRM_DCTWHT_GAPTV('../test_data/Image', '../test_data/Image',img_files);


% plot curves 
cd ..
load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
sav_bytes_save_jpeg{nim} = sav_bytes;
save_ssim_save_jpeg{nim} = sav_ssim;
figure(h8);
subplot(2,4,nim);
plot(sav_bytes,sav_ssim,'k-','LineWidth',2);

h = figure(10); clf;
plot(sav_bytes,sav_ssim,'k-','LineWidth',2);
load(['.\test_data\Image\' filename '.jpeg2000_qual\JPEG2000_qual.mat'])
sav_bytes_save_jpeg2000{nim} = sav_bytes;
save_ssim_save_jpeg2000{nim} = sav_ssim;
hold on
plot(sav_bytes,sav_ssim,'k--','LineWidth',2);

figure(h8);
hold on
plot(sav_bytes,sav_ssim,'k--','LineWidth',2);

for ncs = 1:10
     file_size_cs_WHT(ncs) = stats((ncs-1)*2+1).n_byte;
     cs_ssim_GAP_WHT(ncs) = stats((ncs-1)*2+1).img_ssim; 
     
     file_size_cs_DCT(ncs) = stats((ncs-1)*2+2).n_byte;
     cs_ssim_GAP_DCT(ncs) = stats((ncs-1)*2+2).img_ssim; 
end
figure(h)
hold on;
plot(file_size_cs_WHT,cs_ssim_GAP_WHT,'b--','LineWidth',2);
hold on;
plot(file_size_cs_DCT,cs_ssim_GAP_DCT,'b-','LineWidth',2);

legend('JPEG', 'JPEG2000','CS (GAP-TV), SRM-WHT','CS (GAP-TV), SRM-DCT','Location','southeast');
set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
 title(filename)
saveas(h, ['.\Temp_results\' filename '_JPEG_CS_GAPTV_SRM_WHTDCT.fig']);
saveas(h, ['.\Temp_results\' '_JPEG_CS_GAPTV_SRM_WHTDCT.png']);


 figure(h8)
hold on;
plot(file_size_cs_WHT,cs_ssim_GAP_WHT,'b--','LineWidth',2);
hold on;
plot(file_size_cs_DCT,cs_ssim_GAP_DCT,'b-','LineWidth',2);

set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
 title(filename)
 

stats_save{nim} = stats;
pause(0.1);
end

figure(h8);
legend('JPEG', 'JPEG2000','CS (GAP-TV), SRM-WHT','CS (GAP-TV), SRM-DCT','Location','southeast');

pause(1);
save('.\Temp_results\8image_gap_tv_result_SRM.mat','-v7.3','stats_save','sav_bytes_save_jpeg','save_ssim_save_jpeg',...
     'sav_bytes_save_jpeg2000','save_ssim_save_jpeg2000');

