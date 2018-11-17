function analyze_img(img_dir, fig_num)
  [~,img_name,~] = fileparts(img_dir);
  
  load(fullfile(img_dir, 'results.mat'), 'stats','enc_opts');
  
  figure(fig_num);
  clf;
  
  ylm = [min([stats(:).img_ssim]),max([stats(:).img_ssim])]; %#ok stats read from results.mat
  xlm = [min([stats(:).n_byte]),max([stats(:).n_byte])];
  
  fg_discard = subplot(2,2,1);
  hold on;
  title([img_name ', DISCARD']);
  
  fg_save = subplot(2,2,2);
  hold on;
  title([img_name ', SAVE']);

  fg_narrow = subplot(2,2,3);
  hold on;
  title([img_name ', Narrow Ampl.']);
   
  fg_wide = subplot(2,2,4);
  hold on;
  title([img_name ', Wide Ampl.']);
  
  fg_outrange = {fg_discard, fg_save};
  fg_ampl = {fg_narrow, fg_wide};
  
  q_step = unique([enc_opts(:).qntzr_wdth_mltplr]); %#ok
  q_step = sort(q_step(:),1,'descend'); %#ok
  csr = unique([enc_opts(:).msrmnt_input_ratio]);
  csr = sort(csr); 
  csr_clr = {'b','c','g','m','r'};
  csr_clr = csr_clr(mod((0:(length(csr)-1)),5)+1);
  csr_name = @(k) sprintf('r%03d',csr(k)*1000);
  sns_mtrx = struct('type', unique(arrayfun(@(x) {x.msrmnt_mtrx.type}, enc_opts(:))));
  sns_mtrx_ln = {'x','o'};
  sns_mtrx_name = {'pF','pT'};
  q_ampl = cell(length(sns_mtrx),1);
  for k=1:length(sns_mtrx)
    opts = enc_opts(arrayfun(@(x) strcmp(x.msrmnt_mtrx.type, sns_mtrx(k).type), enc_opts(:)));
    q_ampl{k} = unique([opts(:).qntzr_ampl_stddev]);
    q_ampl{k} = sort(q_ampl{k},'ascend');
  end
  q_ampl_ln = {':','-'};
  q_ampl_ln = q_ampl_ln(mod((0:length(q_ampl_ln)),2)+1);
  q_ampl_name = @(k,m) ['a', num2str(q_ampl{k}(m))];
  q_save = unique([enc_opts(:).qntzr_outrange_action]);
  q_save = sort(q_save, 'ascend');
  q_save_name = {'sF','sT'};
  q_save_ln = {':','-'};
  
%   subplot(1,2,1)
%   hold on
%   title([img_name ', DISCARD']);
%   ylabel('SSIM');
%   xlabel('file size (bytes)');
%   
%   subplot(1,2,2)
%   hold on
%   title([img_name ', save']);
%   ylabel('SSIM');
%   xlabel('file size (bytes)');
% 
  smtrx = [enc_opts(:).msrmnt_mtrx];
  
  for imtrx=1:length(sns_mtrx)
    for isv=1:length(q_save)
      for iampl=1:length(q_ampl{imtrx})
        for iratio=1:length(csr)
          indx = find(...
            strcmp({smtrx(:).type},sns_mtrx(imtrx).type) & ...
            [enc_opts(:).qntzr_outrange_action] == q_save(isv) & ...
            [enc_opts(:).qntzr_ampl_stddev] == q_ampl{imtrx}(iampl) & ...
            [enc_opts(:).msrmnt_input_ratio] == csr(iratio));
          sm = [stats(indx).img_ssim];
          nb = [stats(indx).n_byte];
            
          switch [q_save_ln{isv} q_ampl_ln{iampl}]
            case '--'
              lspc = '-';
            case '::'
              lspc = ':';
            case '-:'
              lspc = '--';
            case ':-'
              lspc = '-.';
          end
          lspc = [csr_clr{iratio} sns_mtrx_ln{imtrx} lspc];   %#ok
          lname = [sns_mtrx_name{imtrx} q_save_name{isv} ...
            q_ampl_name(imtrx,iampl) csr_name(iratio)]; 
          plot(fg_outrange{isv}, nb,sm, lspc, 'DisplayName', lname);
          plot(fg_ampl{iampl},nb,sm, lspc, 'DisplayName', lname);          
        end
      end
    end
  end
  for i=1:2:4
    subplot(2,2,i);
    ylabel('SSIM'); xlabel('file size (bytes)'); xlim(xlm); ylim(ylm);
    lgnd = legend('show','Location','westoutside');
    lgnd.FontSize = 10;
  end    
  for i=2:2:4
    subplot(2,2,i);
    ylabel('SSIM'); xlabel('file size (bytes)'); xlim(xlm); ylim(ylm);
    lgnd = legend('show','Location','eastoutside');
    lgnd.FontSize = 10;
  end    
    
%   for isv=1:length(q_save);
%     figure(fg_outrange{isv});
%     lgnd = legend('show','Location','');
%     lgnd.FontSize = 10;
%   end
%   for iampl=1:size(q_ampl,2)
%     figure(fg_ampl{iampl});
%     lgnd = legend('show','Location','best');
%     lgnd.FontSize = 10;
%   end
%     subplot(1,2,isv);
%     lgnd = legend('show','Location','southeast');
%     lgnd.FontSize = 10;
%     ax = gca();
%     ax.XLim = xlm;
%     ax.YLim = ylm;
%   end
  
end