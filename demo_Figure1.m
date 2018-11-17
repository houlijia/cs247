%main_test
% Just run this code you will see the results of Figure 1 in the main
% paper.
% The image 'Monarch.tif' is used, 2D-WHT is used for sensing matrix and
% D-AMP is used for reconstruction.

% If you want to change images, please simply change the "filename" to one
% of the 8 ncandinates, The true images are stores in ./test_data/Image 

% The output CS files and reconstructed images are stored in
% ./test_data/Image and a folder with the filename

% Xin Yuan & Raziel Haimi-Cohen
% Nokia Bell Labs
% xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 

% For each image, the running time is about 30 minutes, this includes the
% encoding, decoding use D-AMP (much slower than GAP-TV) for reconstruction.

% If you want to see a quick result, please run "demo_firstTry.m"


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
filename = 'Monarch';
img_files= {[filename '.tif']};
[stats, enc_opts] =  run_quant_test_cscd_test_2D_WHT_DAMP('../test_data/Image', '../test_data/Image',img_files);

cd ..
load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
h = figure; 
plot(sav_bytes,sav_ssim,'k->','LineWidth',1);

for ncs = 1:10
     file_size_cs(ncs) = stats(ncs).n_byte;
     cs_ssim_DAMP(ncs) = stats(ncs).img_ssim;  
end
hold on;
plot(file_size_cs,cs_ssim_DAMP,'r-o','LineWidth',1);
title(filename)
legend('JPEG', 'CS (D-AMP), 2D-WHT','Location','southeast');
set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
saveas(h, [filename '_JPEG_CS_DAMP_2DWHT.fig']);
saveas(h, [filename '_JPEG_CS_DAMP_2DWHT.png']);


%plot some images
load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
Num_CS = length(stats);
for nn = 1:Num_CS
    cs_size = stats(nn).n_byte;
    cs_size_err = abs(cs_size - sav_bytes);
    [min_err,ind] = min(cs_size_err);
    if(min_err<100)
    figure; subplot(2,1,1);
    imshow(imread(['.\test_data\Image\' filename '\' filename '-' num2str(nn) '.png']));
    title(['CS (' num2str(cs_size) ' Bytes)'])
    
     subplot(2,1,2);
    if(sav_qual(ind)<10)
        if(mod(sav_qual(ind),1)==0)
           imshow(['.\test_data\Image\' filename '.jpg_qual\' filename '_0' num2str(sav_qual(ind)) '.0.jpg']);
        else
            imshow(['.\test_data\Image\' filename '.jpg_qual\' filename '_0' num2str(sav_qual(ind)) '.jpg']);
        end
    else
      imshow(['.\test_data\Image\' filename '.jpg_qual\' filename '_' num2str(sav_qual(ind)) '.0.jpg']);
    end
    title(['JPEG (' num2str(sav_bytes(ind)) ' Bytes)'])
    end
end