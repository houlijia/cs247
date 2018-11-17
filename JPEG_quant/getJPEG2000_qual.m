function [bytes, j2000_ssim, j2000_psnr, c_rat] = ...
    getJPEG2000_qual( img_file, cmpr_ratio )
  %getJPEG2000_qual returns quality versus file size of JPEG encoding.
  %  
  % INPUT:
  %   img_file: original image file name
  %   cmpr_ratio: (optional) an array of JPEG2000 compression ratios to be used. 
  %                If not present, use def_cmpr_ratio below.
  % OUTPUT:
  %   bytes: array of sizes of the JPEG files in bytes for the given qualities.
  %          The returned sizes are the actual JPEG files size minus 18, to
  %          account for the JFIF metadata section in the files, which is 18
  %          bytes long.
  %   j2000_ssim:  array of SSIM values for the given qualities.
  %   j2000_psnr:  array of PSNR values for the given qualities.
  %   qual:  array of the quality values (should be cmpr_ratio, if supplied).
  
  def_cmpr_ratio = sort([12:2:40 45:5:100 110:10:200 220:20:400 450:50:1000]);
  if nargin < 2
    cmpr_ratio = def_cmpr_ratio;
  end
  cmpr_ratio = cmpr_ratio(:);
    
  [img_dir, img_name, ~] = fileparts(img_file);
  qual_dir = fullfile(img_dir, [img_name '.jpeg2000_qual']);
  
  if ~exist(qual_dir,'dir')
    [ok, err_msg, err_id] = mkdir(qual_dir);
    if ~ok
      error(err_id, 'Failed to create folder %s: %s', qual_dir, err_msg);
    end
  end
  
  qual_file = fullfile(qual_dir, 'JPEG2000_qual.mat');
  
  if isempty(dir(qual_file))
    sav_bytes = [];
    sav_ssim = [];
    sav_psnr = [];
    sav_cmpr_ratio = [];
  else
    load(qual_file, 'sav_bytes', 'sav_ssim', 'sav_psnr', 'sav_cmpr_ratio');
    sav_bytes = sav_bytes(:);                %#ok
    sav_ssim = sav_ssim(:);                  %#ok
    sav_psnr = sav_psnr(:);                  %#ok
    sav_cmpr_ratio = sav_cmpr_ratio(:);      %#ok
    
  end
    
  [c_rat, sav_ind, j2000_ind] = intersect(sav_cmpr_ratio, cmpr_ratio);
  bytes = sav_bytes(sav_ind);
  j2000_ssim = sav_ssim(sav_ind);
  j2000_psnr = sav_psnr(sav_ind);
  
  if length(c_rat) == length(cmpr_ratio)
    return;
  end
  
  I0 = imread(img_file);
  Iorig = im2double(I0);
  
  cmpr_ratio(j2000_ind) = [];
  n = length(cmpr_ratio);
  c_rat = [cmpr_ratio; c_rat];
  sav_cmpr_ratio = [cmpr_ratio; sav_cmpr_ratio];
  bytes = [zeros(n,1); bytes];
  sav_bytes = [zeros(n,1); sav_bytes];
  j2000_ssim = [zeros(n,1); j2000_ssim];
  sav_ssim = [zeros(n,1); sav_ssim];
  j2000_psnr = [zeros(n,1); j2000_psnr];
  sav_psnr = [zeros(n,1); sav_psnr];

  JPEG_IFIF_len = 0;
  
  for i=1:n
    jpg2000_file = fullfile(qual_dir, sprintf('%s_%04.1f.jp2', img_name, c_rat(i)));
    imwrite(I0, jpg2000_file, 'jp2', 'CompressionRatio', c_rat(i));
    fd = dir(jpg2000_file);
    bytes(i) = fd.bytes - JPEG_IFIF_len;
    sav_bytes(i) = bytes(i);
    
    Ijpg = imread(jpg2000_file);
    Ijpg = im2double(Ijpg);
    
    j2000_ssim(i) = ssim(Iorig, Ijpg);
    sav_ssim(i) = j2000_ssim(i);
    
    
    j2000_psnr(i) = psnr(Iorig, Ijpg);
    sav_psnr(i) = j2000_psnr(i);
  end
  
  [sav_cmpr_ratio, indx] = sort(sav_cmpr_ratio);  %#ok
  sav_bytes = sav_bytes(indx);  %#ok
  sav_ssim = sav_ssim(indx);  %#ok
  sav_psnr = sav_psnr(indx);  %#ok
  save(qual_file, 'sav_bytes', 'sav_ssim', 'sav_psnr', 'sav_cmpr_ratio');
  
  [c_rat, indx] = sort(c_rat);
  bytes = bytes(indx);
  j2000_ssim = j2000_ssim(indx);
  j2000_psnr = j2000_psnr(indx);
end
