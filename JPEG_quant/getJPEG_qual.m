function [bytes, jpg_ssim, jpg_psnr, qual] = getJPEG_qual( img_file, jpg_qual )
  %getJPEG_qual returns quality versus file size of JPEG encoding.
  %  
  % INPUT:
  %   img_file: original image file name
  %   jpg_qual: (optional) an array of JPEG quality values to be used. If not
  %             present, use def_qual below.
  % OUTPUT:
  %   bytes: array of sizes of the JPEG files in bytes for the given qualities.
  %          The returned sizes are the actual JPEG files size minus 18, to
  %          account for the JFIF metadata section in the files, which is 18
  %          bytes long.
  %   jpg_ssim:  array of SSIM values for the given qualities.
  %   jpg_psnr:  array of PSNR values for the given qualities.
  %   qual:  array of the quality values (should be jpg_qual, if supplied).
  
  def_qual = [0.5:0.5:10 11:1:25 26:2:32 35:5:50]';
  if nargin < 2
    jpg_qual = def_qual;
  end
  jpg_qual = jpg_qual(:);
  
  [img_dir, img_name, ~] = fileparts(img_file);
  qual_dir = fullfile(img_dir, [img_name '.jpg_qual']);
  
  if ~exist(qual_dir,'dir')
    [ok, err_msg, err_id] = mkdir(qual_dir);
    if ~ok
      error(err_id, 'Failed to create folder %s: %s', qual_dir, err_msg);
    end
  end
  
  qual_file = fullfile(qual_dir, 'JPEG_qual.mat');
  
  if isempty(dir(qual_file))
    sav_bytes = [];
    sav_ssim = [];
    sav_psnr = [];
    sav_qual = [];
  else
    load(qual_file, 'sav_bytes', 'sav_ssim', 'sav_psnr', 'sav_qual');
  end
    
  [qual, sav_ind, jpg_ind] = intersect(sav_qual, jpg_qual);
  bytes = sav_bytes(sav_ind);
  jpg_ssim = sav_ssim(sav_ind);
  jpg_psnr = sav_psnr(sav_ind);
  
  if length(qual) == length(jpg_qual)
    return;
  end
  
  I0 = imread(img_file);
  Iorig = im2double(I0);
  
  jpg_qual(jpg_ind) = [];
  n = length(jpg_qual);
  qual = [jpg_qual; qual];
  sav_qual = [jpg_qual; sav_qual];
  bytes = [zeros(n,1); bytes];
  sav_bytes = [zeros(n,1); sav_bytes];
  jpg_ssim = [zeros(n,1); jpg_ssim];
  sav_ssim = [zeros(n,1); sav_ssim];
  jpg_psnr = [zeros(n,1); jpg_psnr];
  sav_psnr = [zeros(n,1); sav_psnr];

  JPEG_IFIF_len = 18;
  
  for i=1:n
    jpg_file = fullfile(qual_dir, sprintf('%s_%04.1f.jpg', img_name, qual(i)));
    imwrite(I0, jpg_file, 'jpg', 'Quality', qual(i));
    fd = dir(jpg_file);
    bytes(i) = fd.bytes - JPEG_IFIF_len;
    sav_bytes(i) = bytes(i);
    
    Ijpg = imread(jpg_file);
    Ijpg = im2double(Ijpg);
    
    jpg_ssim(i) = ssim(Iorig, Ijpg);
    sav_ssim(i) = jpg_ssim(i);
    
    
    jpg_psnr(i) = psnr(Iorig, Ijpg);
    sav_psnr(i) = jpg_psnr(i);
  end
  
  [sav_qual, indx] = sort(sav_qual);  %#ok
  sav_bytes = sav_bytes(indx);  %#ok
  sav_ssim = sav_ssim(indx);  %#ok
  sav_psnr = sav_psnr(indx);  %#ok
  save(qual_file, 'sav_bytes', 'sav_ssim', 'sav_psnr', 'sav_qual');
  
  [qual, indx] = sort(qual);
  bytes = bytes(indx);
  jpg_ssim = jpg_ssim(indx);
  jpg_psnr = jpg_psnr(indx);
end
