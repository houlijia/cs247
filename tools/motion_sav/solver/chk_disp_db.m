function [ok, psnr, psnr_list ] = chk_disp_db(xvec, cmp_blk_psnr, psnr_list)
  psnr = [];
  ok = false;
  if isempty(cmp_blk_psnr) && isempty(psnr_list)
    return
  end
  
  psnr = cmp_blk_psnr(xvec);
  indc = find(psnr >= psnr_list);
  if ~isempty(indc)
    ok = true;
    psnr_list(indc) = [];
  end
end

