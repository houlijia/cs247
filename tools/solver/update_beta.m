function [beta, changed] = update_beta(beta, opts, cnstrnts_met, do_update)
  if nargin < 3
    do_update = true;
  end
  if do_update
    prev_beta = beta;
    if ~cnstrnts_met.D
      beta.D = beta.D*opts.beta_rate;
    end
    if ~cnstrnts_met.A
      beta.A = beta.A*opts.beta_rate;
    end
    beta = restrict_beta(beta);
    changed = ~isequal(beta, prev_beta);
  else
    changed = false;
  end
end

