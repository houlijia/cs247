function [stats, enc_opts] = test_multimatrix(img_inp_dir, img_out_dir, prll)
  % Run a set of tests on various cases 
  if nargin < 3
    prll = false;
  end
  
  img_out_dir = regexprep(img_out_dir, '*',...
      datestr(now,'yyyymmdd_HHMM'),'once');
  img_files = {...
    'barbara.tif', 'boats.tif',...
    'cameraman.tif', 'foreman.tif', 'house.tif', 'lena256.tif', ...
    'Monarch.tif', 'Parrots.tif', 'trui1.png' };
  img_files = arrayfun(@(x) {fullfile(img_inp_dir, x{1})}, img_files);
  
  %   q_step = [0.5 1 1.5 2 3 4];
  q_step = 2;
  csr = [0.02:0.01:0.05, 0.06:0.02:0.10, 0.15:0.05:0.25];
%   min_row_ratio = {[0.01,0], [0.02,0], [0.03,0], [0.04,0], [0.05,0] };
  min_row_ratio = [0,0];
  row_share = {[4,0], [3,1], [2,2], [1,3], [0,4]};
  
  mtrx1 = struct('type', { 'MD_WH', 'MD_DCT' });
  mtrx2 = struct('type', 'WH', 'args', struct('nodc', true));
  mtrx2 = struct('type', 'Combine', 'args', struct(...
    'wg', {{ 0.2, 0.5, 1, 2 }}, 'mtrcs__', {{ mtrx2 }} ));
  enc_opts = struct(...
    'msrmnt_input_ratio', {num2cell(csr)}, ...
    'qntzr_wdth_mode', CS_EncParams.Q_WDTH_CSR, ...
    'qntzr_wdth_mltplr', {num2cell(q_step)}, ...
    'qntzr_ampl_stddev', 4,...
    'qntzr_outrange_action', CS_EncImgParams.Q_SAVE,...
    'msrmnt_mtrx', struct('type', 'Concat', ...
       'args', struct(...
         'mtrcs_', struct('mtrcs__', {{ mtrx1(1), mtrx2}, {mtrx1(2), mtrx2}}), ... {{{ mtrx1(1), mtrx2}, {mtrx1(2), mtrx2}}},...
         'min_row_ratio', {min_row_ratio},...
         'row_share', {row_share},...
         'normalize', 1))...       
    );
  
  stats = test_quant_set(img_files, enc_opts, 1, img_out_dir, prll);
end
              
          
  
