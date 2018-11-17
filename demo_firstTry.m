%main_test
% Just run this code you will see the results

% If you want to change images, please simply change the "filename" to one
% of the 8 ncandinates, The true images are stores in ./test_data/Image 

% The output CS files and reconstructed images are stored in
% ./test_data/Image and a folder with the filename

% Xin Yuan & Raziel Haimi-Cohen
% Nokia Bell Labs
% xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 

% For each image, the running time is about 3 minutes, this includes the
% encoding, decoding use GAP-TV for reconstruction.

clear all
close all
clc



cd ./tools/scripts
set_mex_path dbg-ng vc
cd ..
cd ..
addpath(genpath('../CS_JPEG_demo'));
%%

for nim = 4
    
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
[stats, enc_opts] =  run_quant_test_cscd_test_2D_DCT('../test_data/Image', '../test_data/Image',img_files);


% plot curves 
cd ..
load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
sav_bytes_save_jpeg{nim} = sav_bytes;
save_ssim_save_jpeg{nim} = sav_ssim;

h = figure(10); clf;
plot(sav_bytes,sav_ssim,'k-','LineWidth',2);
load(['.\test_data\Image\' filename '.jpeg2000_qual\JPEG2000_qual.mat'])
sav_bytes_save_jpeg2000{nim} = sav_bytes;
save_ssim_save_jpeg2000{nim} = sav_ssim;
hold on
plot(sav_bytes,sav_ssim,'k--','LineWidth',2);


for ncs = 1:10
     file_size_cs_DCT(ncs) = stats(ncs).n_byte;
     cs_ssim_GAP_DCT(ncs) = stats(ncs).img_ssim; 
end
figure(h)
hold on;
plot(file_size_cs_DCT,cs_ssim_GAP_DCT,'b-','LineWidth',2);

legend('JPEG', 'JPEG2000','CS (GAP-TV), 2D-DCT','Location','southeast');
set(gca,'FontSize',12)
xlabel('Compressed file size (Bytes)','FontSize',12)
ylabel('Structural Similarity (SSIM)','FontSize',12)
 title(filename)
saveas(h, [filename '_JPEG_CS_GAPTV_2D_DCT.fig']);
saveas(h, [filename '_JPEG_CS_GAPTV_2D_DCT.png']);



load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
Num_CS = length(stats);
for nn = 1:Num_CS
    cs_size = stats(nn).n_byte;
    cs_size_err = abs(cs_size - sav_bytes);
    [min_err,ind] = min(cs_size_err);
    if(min_err<60)
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


end

