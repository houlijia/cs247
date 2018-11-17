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
      beta.scldA = beta.scldA*opts.beta_rate;
    end
    beta = restrict_beta(beta);
    changed = struct('D',(beta.D ~= prev_beta.D), 'A',(beta.A ~= prev_beta.A));
  else
    changed = struct('D',false, 'A',false);
  end
end

