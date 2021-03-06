function analyze_all_img_hist_type(out_dir, img_num, cnstrnts, mode, lims, orig_dir,...
    show_jpg2000)
  % analyze_all_img analyzes statistics files generated by run_quant_test and
  % test_quant_set.
  %
  % The program generates a Matlab figure for each processed image
  % INPUT:
  %   out_dir: the output directory, where for each picture
  %             there is a subdirectory with picture names. 
  %   img_num: If a non-cell scalar, figure number for first picture, incremented
  %       sequentially, otherwise an array of figure numbers or figure handles
  %       to use for each picture.
  %   cnstrnts: a struct or array of structs sepcifying which type of results to
  %             plot (only results that match at least one struct in cnstrnts).
  %             To specify no constraints enter struct().
  %   mode: 1 - SSIM ver. number of byte, parameterized by quantization step
  %             (default).
  %         2 - SSIM ver. number of byte, parameterized by compression ratio
  %         3 - Msrs. quantization error RMS to msrs RMS ratio (db). ver. number
  %             of bits per used measurement, parameterized by quantization step
  %  lims:  if present a struct with optional fields xlim and/or ylim, which
  %         specify the range of the figures. Otherwise the range is determined
  %         automatically.
  %  orig_dir: (optional): The directory where the original picture is. If
  %             specified, JPEG performance is also drawn.
  %  show_jpg2000: If true and orig_dir is present, show also JPEG2000 performance.
  
  if nargin < 3
    cnstrnts = struct();
  end
  if nargin < 4
    mode = 1;
  end
  if nargin < 5
    lims = struct();
  end
  if nargin < 6
    orig_dir = '';
  end
  if nargin < 7
    show_jpg2000 = ~isempty(orig_dir);
  end
  
  dr = dir(out_dir);
  
  img_indx = 0;
  for k=1:length(dr)
    if ~dr(k).isdir || any(strcmp(dr(k).name, {'.','..'}))
      continue;
    end
    
    img_indx = img_indx+1;
    if iscell(img_num)
      img_id = img_num{img_indx};
    elseif isscalar(img_num)
      img_id = img_num + img_indx - 1;
    else
      img_id = img_num(img_indx);
    end
    img_dir = fullfile(out_dir, dr(k).name);
    switch(mode)
      case 1
        analyze_img_slct_hist_type(img_dir,  img_id, cnstrnts, true, lims, orig_dir, show_jpg2000);
      case 2
        analyze_img_slct_hist_type(img_dir,  img_id, cnstrnts, false, lims, orig_dir, show_jpg2000);
      case 3
        analyze_qmsr_slct(img_dir,  img_id, cnstrnts, lims);
      otherwise
        error('Unexpected mode: %d', mode);
    end
  end
end