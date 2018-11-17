function [prefix, cmp_blk_psnr] = parse_solver_proc_params(proc_params)
  if isfield(proc_params, 'prefix')
    prefix = proc_params.prefix;
  else
    prefix = '';
  end
  if isfield(proc_params, 'cmp_blk_psnr')
    cmp_blk_psnr = proc_params.cmp_blk_psnr;
  else
    cmp_blk_psnr = [];
  end
end

