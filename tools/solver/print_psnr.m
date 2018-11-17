function [ disp_db, disp_psnr ] = print_psnr(prfx, do_disp, xvec, cmp_blk_psnr, disp_db,...
    thresh, primal_chg, dual_chg)
  %print_psnr - print PSNR and primal/dual change if necessary
  % INPUT:
  %   do_disp - If true, displaying is necessary
  %   xvec - the current solution vector
  %   cmp_blk_psnr - a function which computes PSNR using xvec
  %   disp_db - A list of PSNR values that when reachec, a PSNR value
  %             should be computed
  %   thresh - threshold to compare primal and dual change to
  %   primal_chg - primal change
  %   dual_chg - dual cnage(optional)
  
  
  if do_disp || ~isempty(disp_db)
    psnr = [];
    disp_psnr = false;
    if ~isempty(cmp_blk_psnr) && ~isempty(disp_db)
      psnr = cmp_blk_psnr(xvec);
      indc = find(psnr >= disp_db);
      if ~isempty(indc)
        disp_psnr = true;
        disp_db(indc) = [];
      end
    end
  else
    disp_psnr = false;
  end
  
  if disp_psnr || do_disp
    if nargin >=8
      has_dual = true;
      passed = (primal_chg < thresh) && (dual_chg < thresh);
    else
      has_dual = false;
      passed = false;
    end
    
    if passed
      ok_str = 'PASSED';
    else
      ok_str = 'NOT passed';
    end
    if isempty(psnr) && ~isempty(cmp_blk_psnr)
      psnr = cmp_blk_psnr(xvec);
    end
    if ~isempty(psnr)
      ok_str = sprintf('PSNR=%4.1f, %s', psnr, ok_str);
    end
    if has_dual
      fprintf('%s lgrng_chg: [%10.3g, %10.3g] (%g) %s\n', prfx(),...
        primal_chg, dual_chg, thresh, ok_str);
    else
      fprintf('%s lgrng_chg: %10.3g (%g)               %s\n', prfx(),...
        primal_chg, thresh, ok_str);
    end
  end
  
  
end

