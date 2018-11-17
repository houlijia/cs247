% Just run this code you will see the results without quantization



% Xin Yuan & Raziel Haimi-Cohen
% Nokia Bell Labs
% xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 



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

CSr_all_ini = [0.02:0.01:0.05, 0.06:0.02:0.10, 0.15:0.05:0.25];
algo_all = {'GAP_TV','DAMP','NLR_CS'}; % algorithms used

h8 = figure(8);


for nim =1:8
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
%filename = 'boats.tif';
image_file = ['../test_data/Image/' filename '.tif'];
I0 = imread(image_file);

figure(h8);
subplot(2,4,nim);

for ncs = 1:length(CSr_all_ini)
    
    CSr = CSr_all_ini(ncs);    
    for nf = 1:2
      switch nf
          case 1 
              enc_opts = struct(...
                'msrmnt_input_ratio', CSr, ...
                'msrmnt_mtrx', struct('type', 'MD_WH','nxt',{[]}));
            
            disp(['Image: ' filename ', CSr: ' num2str(CSr) ', Sensing matrix: 2D-WHT']);
          case 2
              enc_opts = struct(...
                'msrmnt_input_ratio', CSr, ...
                'msrmnt_mtrx', struct('type', 'MD_DCT','nxt',{[]}));
            disp(['Image: ' filename ', CSr: ' num2str(CSr) ', Sensing matrix: 2D-DCT']);
       end

    [orig_img, y, coding_info] = test_noquan(...
        image_file, enc_opts);

       for nAlg = 1:length(algo_all)
            % Reconstruct from quantized measurements
             clear dec_opts
             dec_opts = struct('alg', algo_all(nAlg));
            
             [rec_q, rec_q_raw] = CSImgReconstruct(y, coding_info, dec_opts);
             im_rec{nim,ncs,nAlg,nf} = uint8(round(rec_q_raw));
             
             PSNR_all(nim,ncs, nAlg,nf) = psnr(im_rec{nim,ncs,nAlg,nf}, I0);
             SSIM_all(nim,ncs, nAlg,nf) = ssim(im_rec{nim,ncs,nAlg,nf}, I0);
             disp([algo_all(nAlg) ' done...']);
        end     
% 
    end

end
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 1,1)),'b--');
    hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 2,1)),'g--');
     hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 3,1)),'r--');
    hold on
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 1,2)),'b-');
    hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 2,2)),'g-');
     hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 3,2)),'r-');
     set(gca,'FontSize',12)
    xlabel('CSr','FontSize',12)
    ylabel('Structural Similarity (SSIM)','FontSize',12)
     title(filename);
    
    h = figure(1); clf;
    
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 1,1)),'b--');
    hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 2,1)),'g--');
     hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 3,1)),'r--');
    hold on
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 1,2)),'b-');
    hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 2,2)),'g-');
     hold on;
    plot(CSr_all_ini,squeeze(SSIM_all(nim,:, 3,2)),'r-');
    set(gca,'FontSize',12)
    xlabel('CSr','FontSize',12)
    ylabel('Structural Similarity (SSIM)','FontSize',12)
     title(filename);
      legend( 'CS (GAP-TV), 2D-WHT','CS (D-AMP), 2D-WHT','CS (NLR-CS), 2D-WHT',...
    'CS (GAP-TV), 2D-DCT','CS (D-AMP), 2D-DCT','CS (NLR-CS), 2D-DCT','Location','southeast');
saveas(h, ['..\Temp_results\' filename '_noQuan_CS_3Algo_2DDCTWHT.fig']);
saveas(h, ['..\Temp_results\' filename '_noQuan_CS_3Algo_2DDCTWHT.png']);
     
    pause(0.1);
end
figure(h8);
  legend( 'CS (GAP-TV), 2D-WHT','CS (D-AMP), 2D-WHT','CS (NLR-CS), 2D-WHT',...
    'CS (GAP-TV), 2D-DCT','CS (D-AMP), 2D-DCT','CS (NLR-CS), 2D-DCT','Location','southeast');

save(['..\Temp_results\' filename '_noQuan_CS_all3_2D_DCTWHTsensing.mat'], '-v7.3','im_rec','PSNR_all','SSIM_all');

saveas(h8, ['..\Temp_results\8img_noQuan_CS_3Algo_2DDCTWHT.fig']);
saveas(h8, ['..\Temp_results\8img_noQuan_CS_3Algo_2DDCTWHT.png']);
     
%save_data_dwt = save_data;
% 
% load(['.\test_data\Image\' filename '.jpg_qual\JPEG_qual.mat'])
% ssim_cs = cell2mat(save_data.Rec_ssim_DAMP);
% file_size_cs = cell2mat(save_data.cs_filesize);
% figure; plot(file_size_cs,ssim_cs);
% hold on
% plot(sav_bytes,sav_ssim)
% legend('CS', 'JPEG')
% hold on
% ssim_cs = cell2mat(save_data.Rec_ssim);
% file_size_cs = cell2mat(save_data.cs_filesize);
% plot(file_size_cs ,ssim_cs);
% legend('CS, dct sensing matrix', 'JPEG','CS, dwt sensing matrix');


