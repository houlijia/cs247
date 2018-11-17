

% demo_Figure4
% Just run this code you will see the results of Figure 4 in the main
% paper.
% All the 8 images are used. Both 2D-WHT and 2D-DCT in zig-zag order are used for sensing matrix
% GAP-TV, D-AMP adn NLR-CS are used for reconstruction.
% therefore, it include 160 encoding times (8 images x 2 sensing matrices x 10 CSr)
% 480 reconstructions: 160 x 3 algorithms


% The output CS files and reconstructed images are stored in
% ./test_data/Image and a folder with the filename

% Xin Yuan & Raziel Haimi-Cohen
% Nokia Bell Labs
% xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 

% For each image, the running time is about 70 minutes

% If you want to see a quick result of all the eight images, please run "demo_GAPTV.m"


clear all
close all
clc



cd ./tools/scripts
set_mex_path dbg-ng vc
cd ..
cd ..
addpath(genpath('../CS_JPEG_demo'));
%addpath(genpath('../.. ./CS_jpeg_v2'));
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
%filename = 'boats'; %Monarch, foreman,boats, lena256, barbara,  house, cameraman, Parrots
img_files= {[filename '.tif']};
[stats, enc_opts] =  run_quant_test_cscd_test_2D_DCTWHT_3Algo('../test_data/Image', '../test_data/Image',img_files);


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
     file_size_cs_WHT(ncs) = stats((ncs-1)*2+1,1).n_byte;
     cs_ssim_GAP_WHT(ncs) = stats((ncs-1)*2+1,1).img_ssim;  
     cs_ssim_DAMP_WHT(ncs) = stats((ncs-1)*2+1,2).img_ssim;  
     cs_ssim_NLR_WHT(ncs) = stats((ncs-1)*2+1,3).img_ssim; 
     
     file_size_cs_DCT(ncs) = stats((ncs-1)*2+2,1).n_byte;
     cs_ssim_GAP_DCT(ncs) = stats((ncs-1)*2+2,1).img_ssim;  
     cs_ssim_DAMP_DCT(ncs) = stats((ncs-1)*2+2,2).img_ssim;  
     cs_ssim_NLR_DCT(ncs) = stats((ncs-1)*2+2,3).img_ssim; 
end
hold on;
plot(file_size_cs_DCT,cs_ssim_GAP_DCT,'b-','LineWidth',2);
hold on;
plot(file_size_cs_DCT,cs_ssim_DAMP_DCT,'g-','LineWidth',2);
hold on;
plot(file_size_cs_DCT,cs_ssim_NLR_DCT,'r-','LineWidth',2);

hold on;
plot(file_size_cs_WHT,cs_ssim_GAP_WHT,'b--','LineWidth',2);
hold on;
plot(file_size_cs_WHT,cs_ssim_DAMP_WHT,'g--','LineWidth',2);
hold on;
plot(file_size_cs_WHT,cs_ssim_NLR_WHT,'r--','LineWidth',2);

title(filename)

set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
 title(filename)
 
 
figure(h)
hold on;
plot(file_size_cs_DCT,cs_ssim_GAP_DCT,'b-','LineWidth',2);
hold on;
plot(file_size_cs_DCT,cs_ssim_DAMP_DCT,'g-','LineWidth',2);
hold on;
plot(file_size_cs_DCT,cs_ssim_NLR_DCT,'r-','LineWidth',2);

hold on;
plot(file_size_cs_WHT,cs_ssim_GAP_WHT,'b--','LineWidth',2);
hold on;
plot(file_size_cs_WHT,cs_ssim_DAMP_WHT,'g--','LineWidth',2);
hold on;
plot(file_size_cs_WHT,cs_ssim_NLR_WHT,'r--','LineWidth',2);

title(filename)

set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
 title(filename)
 
legend('JPEG', 'JPEG2000','CS (GAP-TV), 2D-DCT','CS (D-AMP), 2D-DCT','CS (NLR-CS), 2D-DCT',...
    'CS (GAP-TV), 2D-WHT','CS (D-AMP), 2D-WHT','CS (NLR-CS), 2D-WHT','Location','southeast');
saveas(h, [filename '_JPEG_CS_3Algo_2DDCTWHT.fig']);
saveas(h, [filename '_JPEG_CS_3Algo_2DDCTWHT.png']);


stats_save{nim} = stats;

end

figure(h8);
legend('JPEG', 'JPEG2000','CS (GAP-TV), 2D-DCT','CS (D-AMP), 2D-DCT','CS (NLR-CS), 2D-DCT',...
    'CS (GAP-TV), 2D-WHT','CS (D-AMP), 2D-WHT','CS (NLR-CS), 2D-WHT','Location','southeast');

save('8image_All3_2DDCTWHT_result.mat','-v7.3','stats_save','sav_bytes_save_jpeg','save_ssim_save_jpeg',...
    'sav_bytes_save_jpeg2000','save_ssim_save_jpeg2000');
