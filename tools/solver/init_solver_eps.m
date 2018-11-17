% init_solver_eps - initialize the epsilons for the solver
%   Detailed explanation goes here
function eps  = init_solver_eps( opts, sparser, b, q_maxerr, q_stdv_err)
    n_b = length(b);
    eps = struct(...
      'lgrng_chg', max(opts.eps_lgrng_chg_init,opts.eps_lgrng_chg),...
      'lgrng_chg_final', opts.eps_lgrng_chg,...
      'lgrng_chg_rate', opts.eps_lgrng_chg_rate...
      );

    eps.A_maxerr = q_maxerr + opts.eps_A_maxerr * q_stdv_err;
    eps.A_sqrerr = opts.eps_A_sqrerr * sqrt(n_b * (q_stdv_err^2 +...
      q_maxerr^2/3)); 
    eps.D_maxerr = opts.eps_D_maxerr;
    eps.D_sqrerr = opts.eps_D_sqrerr * sqrt(sparser.dimSprsVec());
    
end
