% solveQuant solves the Compressive sensing reconstruction problem for
% quantized measurements.
%
% Goal: Find a raw data vector x, s.t. |Ax-b|<q where b is a measurements
% vector, order(b) < order(x), q is a maximal quantization error (a scalar)
% and the inequality applies to each component of Ax-b.  We are given a
% transform w=D(x) which sparsifies x (the order of w may be different from
% the order of x, higher or lower).  Using the sparsity we attempt to find
% x (or an approximation to it) by minimizing a target function J(w) (e.g.
% an L_1 norm of w).:
%
%      min_{x,w} J(w) s.t. Dx=w, |Ax-b|<=q
%
% Note that J and D are provided to this function only implicitly as
% methods of the input argument object "sparser".
%
% This function is an extension of solveExact(), so some of the common
% details may be elaborated only in the documentation of that function.
%
% Let 
%     p(t) = min(t+q, max(t-q, 0))
% That is, p(t)=0 if |t|<=q, p(t)=t-q if t>q and p(t)=t+q if t<q. Note that 
%     p`(t) = true(p(t))
% where true(s) is 0 if s=0 and 1 otherwise. Note that p(t)*p`(t)=p(t).
%
% If t is a vector let p(t) and p`(t) be defined as vectors the components
% of which are the results of applying of p() or p`(), respectively, to the 
% corresponding components of the t.
%
% The constraint |Ax-b|<=q is equaivalent to p(Ax-b)=0.
% The constrained minimization is done by the minimizing the augmented
% Lagrangian function:
%
% min_{x,w} l(x,w) = J(w) + lambda_D'(Dx-w) + (beta_D/2)||Dx-w||_2^2
%                 + lambda_A'p(Ax-b) + (beta_A/2)||p(Ax-b)||_2^2
%                 s.t. Dx=w p(Ax-b)=0
%
% where lambda_D and lambda_A are Lagrange multipliers. beta_A and beta_D are
% penalty costs which are initialized to beta_initial_A and beta_inital_D,
% respectively.
%
% The solution follows the standard augmented Lagrangian approach
% where in each iteration we find the minimum of l(x,w) given a particular
% set of Lagrange multipliers and then update the multipliers using the
% equations:
%    lambda_A <- lambda_A + beta_A*p(Ax-b)
%    lambda_D <- lambda_D + beta_D*(Dx-w)
% This constitutes the outer loop.  During the iterations the values of
% beta_A, beta_D are increased until reaching a final value:
%
% The calculation of x and w is done
% in an ineternal loop which alternatiely minimizes l(x,w) with respect to
% x given w and then with respect to w given x.  
%
% The sensing matrix and the sparser are given as input arguments and are
% expected to be objects with methods. The minimization of w
% given x is done by a method of the sparser and is
% not a part of this function description. The minimization of x given w
% is done by running a few steps of congjugate gradient. This is done using
% the equations for the gradient and the Hessian of the Lagrangian:
%
%    g_x(x)' = grad_x l(x,w)' =  D'lambda_D + A'(lambda_A.*p`(Ax-b))
%                 + beta_D D'(Dx-w) + beta_A A'(p(Ax-b).*p`(Ax-b))
%             
%   =  D'lambda_D + beta_D D'(Dx-w) + A'((lamdba_A + beta_A*p(Ax-b)).*p`(A))
%   =  D'lambda_D + beta_D D'(Dx-w) + A'((lamdba_A.*p`(Ax-b)) + beta_A*A'*p(Ax-b))
%
% where .* indicates entry by entry multiplication.
% Note that the first summand is constant w.r.t. x.
%
% Let
%
%       L = Hessian_x (x,w) = beta_D D'D + beta_A A'[p`(Ax-b).*A]
%         = beta_D D'D + beta_A A' diag(p`(Ax-b) A
%
% where [p`(Ax-b).*A] ia the matrix obtained by multiplying each row of A
% by the corresponding value of p`(Ax-b) and diag(p`(Ax-b) is the diagonal
% matrix with p`(Ax-g) as the diagonal entries, that is, the i-th entry is
% true(|(Ax-b)_i| > q).
%
%      h(d,e) = d'Le = beta_D dot(D d,D e) + 
%                      beta_A dot(p`(Ax-b).*A d, p`(Ax-b).*A e));
%
%  The last dot product means that after computing A d and A e we should
%  set to zero the entries for which p(Ax-b) is zero and then perform the
%  dot product.
%
%  The conjugate gradient start with an inital guess x as
%  follows:
%    Initialize: g = g_x(x), d=-g
%     Compute D'g, A'g and h(g,g)
%                set d = - g, D'd = -D'g, A'd = -A'g, h(d,d) = h(g,g)
%    Repeat n_A times:
%         s <- ((g'd)/h(d,d))  << compute step size
%         if ||s|| < threshold break << check for convergence 
%         x  <-  x + s*d  << Update x
%         g  <- g + (s*beta_A)*(A'*A*d) + (s*beta_D)*(D'*D*d) << new grad_x
%         d  <- -g + (h(g,d)/h(d,d))d  << update search direction
%  Unlike the exact case, this is not a quadratic minimization problem,
%  since the Hessian is not constant and the gradient does not change
%  linearly as function of x.  Therefore, it is possible that a step will
%  result in increasing, rather than decresing the object function. Hence 
%  after each update of x we check if the Lagrangian was actually reduced. 
%  If it was not, the conjugate gradient is restarted (thus the direction 
%  is minus the gradient) and the step size is determined by line search.
%     CHECK IF RESTARTING NEEDED
%
% A note about beta and sens_mtrx: multiplication by the matrix and its
% transpose are preformed by functions of the object sens_mtrx.  Somtimes
% the transpose function actually scales the output by a constant. This is
% equivalent to scaling beta by the same constant.  Therefore, in
% multiplications involving beta which do not involve multiplication by the
% transpose we multiply by a scaled version of beta.
%
% variables in this code:
%
% xerr.A = Ax-b
% xerr.D = Dx-w
% xerr.J = J(w)
% sqrerr.A = ||w - Dx||^2
% sqrerr.D= ||Ax -b ||^2
% cnj_err.A  = A'(Ax-b)
% cnj_err.D = D'(Dx-w) (coefficients beta and mu are not included)
% lgr = l(x,w) =
%       J(w) + beta.D/2 sqrerr.D + beta.A/2 sqrerr.A
%                  + lambda.D` xerr.D + lambda.A' xerr.A
%
%
% Written by Raziel Haimi-Cohen, Bell-Labs, in Jan. 2012, based on an
% earlier version by Chengbo Li @ Bell Laboratories, Alcatel-Lucent
% Computational and Applied Mathematics department, Rice University
% 06/12/2010
%
%  Input arguments
%    sens_mtrx - sensing matrix
%    b - measurements vector,
%    sparser - sparsifier,
%    opts - an object of type CS_DecParams, specifying decoding parameters
%    q_step - quantizer step size
%    msrs_pxl_stdv_err - Standard deviation of measurements error caused by
%      errors in pixels.
%    proc_params - (optional) a struct which may contain additional
%      parameters: recognized fields:
%        prefix - prefix to use (default '')
%        cmp_blk_psnr - a function pointer to a function which computes PSNR
%                       of this block (empty = none)
%  Output arguments
%    xvec - reconstructed vector
%    done - true if reconstruction was successful
%    lambda - the lagrange coefficients of the optimal solution
%    beta - beta values in the optimized solution
%    out - statistics computed if opts.cmp_solve_stats is true. Otherwise
%          [] is returned.

function [xvec, done, lambda, beta, out] = ...
    solveQuant(sens_mtrx, b, sparser, opts, q_step, msrs_pxl_stdv_err, proc_params)
 
  if nargin < 7
    proc_params = struct();
  end
  [prefix, cmp_blk_psnr] = parse_solver_proc_params(proc_params);
            
  % initialize xvec (x), beta, eps
  eps = init_solver_eps(opts, sparser, b,...
    0, msrs_pxl_stdv_err);
  maxAerr = q_step/2;
  
  sparser.reset();
  
  if opts.solve_exact_fctr
    opts_e = opts.copy();
    opts_e.init = 0;
    opts_e.eps_lgrng_chg = opts_e.eps_lgrng_chg * opts.solve_exact_fctr;
    opts_e.eps_D_maxerr = opts.eps_D_maxerr * opts.solve_exact_fctr;
    opts_e.eps_D_sqrerr = opts.eps_D_sqrerr * opts.solve_exact_fctr;
    opts_e.eps_A_sqrerr = opts.eps_A_maxerr * opts.solve_exact_fctr;
    opts_e.eps_A_sqrerr = opts.eps_A_sqrerr * opts.solve_exact_fctr;
    proc_params.prefix = [prefix 'ex '];
    [xvec, ~, lambda] =...
      solveExact(sens_mtrx, b, sparser, opts_e, opts.solve_exact_fctr*q_step, ...
      msrs_pxl_stdv_err, proc_params);
  else
    xvec = init_solver_xvec(opts, sens_mtrx, b);
    lambda = struct('D',sens_mtrx.zeros(sparser.n_sprsvec,1),...
      'A',sens_mtrx.zeros(length(b),1));
  end
  disp_db = opts.disp_db;
  
  beta = init_solver_beta(opts, sens_mtrx, sparser);
  
  xerr = struct('A0', sens_mtrx.multVec(xvec) -b,...
    'maxAerr', maxAerr, 'wdthAerr', 0,'A',[],'D', [], 'J', 0);
  xerr0 = xerr;
  xerr.wdthAerr = min(xerr.maxAerr, ...
    opts.q_trans_Aerr*maxAerr + opts.q_trans_msrs_err*msrs_pxl_stdv_err);
%   xerr.wdthAerr = 0;
  xerr.A = comp_p(xerr.A0, maxAerr, xerr.wdthAerr);
  grad_x_const = sparser.compSprsVecTrnsp(lambda.D);
  [xerr.D, xerr.J] = sparser.optimize(xvec, lambda.D, beta.D);
  [grad_x, Agrad_x, pAgrad_x, Dgrad_x, ~, gLg] = ...
    compGrad_x(xerr, grad_x_const, beta, lambda, sens_mtrx, sparser);
  [lgrngn,sqrerr] = compLgrngn(beta, lambda, xerr);
  
  iter = struct(...
    'E',0,...  % External iterations count
    'I',0, ...  % Internal iterations count in this external iteration
    'C',0, ... % Internal iterations count in previous iterations
    'L',0 ... % Line searches
    );
  
  if opts.cmp_solve_stats
    out_cnt = 1;
    out = setStatus(beta, eps, iter, lgrngn, sqrerr, 0, 0, xerr, 0, 0);
  else
    out_cnt = 0;
    out = [];
  end
  
  function str = prfx()
    str = sprintf('%s%d(%d:%d:%d) ', prefix, iter.E, iter.I,iter.C,iter.L);
  end
  
  done = false;
  for ext_iter = 1:opts.max_out_iters
    if(iter.C >= opts.max_iters)
      break;
    end
    iter.E = ext_iter;
    iter.C = iter.C + iter.I;
    iter.I = 0;
    
    do_disp = opts.disp>0 && (mod(iter.E,opts.disp) == 0);
    if do_disp
      fprintf('%sSolveQuant Starting iteration\n', prfx());
    end
    prev_lgrngn = lgrngn;
    
    % Run internal iterations - optimize x and w given lambda and beta
    init_cg = true;
    for int_iter = 1:min(opts.max_int_iters, opts.max_iters-iter.C)
      iter.I = int_iter;
      
      if ~init_cg
        gLd = beta.D * dot(Dgrad_x, Ddrct) + beta.A*dot(pAgrad_x,Adrct);
        past_wgt = gLd/dLd;
        dLd = gLg  + past_wgt*(-2*gLd + past_wgt*dLd);
        
        % Update the direction drct based on the new gradient
        drct = - grad_x + past_wgt*drct;
        Adrct = -Agrad_x + past_wgt*Adrct;
        Ddrct = -Dgrad_x + past_wgt*Ddrct;
      else
        past_wgt = 0;
        dLd = gLg;
        drct = -grad_x;
        Adrct = -Agrad_x;
        Ddrct = -Dgrad_x;
        init_cg = false;
      end
      
      step_size = - dot(drct, grad_x)/dLd;
      
      lgr = lgrngn; % last inner iteration lagrangian
      if iter.I > 1
        pdlgr = dlgr;
      end
      
      [lgrngn1, xvec1, xerr1] =...
         move_step(xvec, xerr, drct, Adrct, Ddrct, step_size, lambda, beta);
      
      if lgrngn1 >= lgrngn
        if past_wgt
          % Try the gradient direction
          dLd = gLg;
          drct = -grad_x;
          Adrct = -Agrad_x;
          Ddrct = -Dgrad_x;
          step_size = - dot(drct, grad_x)/dLd;
          [lgrngn1, xvec1, xerr1] =...
            move_step(xvec, xerr, drct, Adrct, Ddrct, step_size, lambda, beta);
          iter.L = iter.L+1;
        end
        
        % Binary search
        if lgrngn1 >= lgrngn
          init_cg = true;
          ee = find_hessian_changes(xerr, Adrct, step_size);
          % - Binary search on ee
          k = length(ee);
          while k > 0 && lgrngn1 >= lgrngn
            k = floor(k/2);
            if k > 0
              step_size = ee(k);
            else
              step_size = ee(1)/2;
            end
            [lgrngn1, xvec1, xerr1] =...
              move_step(xvec, xerr, drct, Adrct, Ddrct, step_size,...
              lambda, beta);
            iter.L = iter.L+1;
          end
        end
      end
      
      if lgrngn1 >= lgrngn
        iter.I = iter.I-1;
        break;
      end
      
      lgrngn = lgrngn1; xvec = xvec1; xerr = xerr1;
      dlgr = lgr - lgrngn;
      
      % Update gradient
      [grad_x, Agrad_x, pAgrad_x, Dgrad_x, ~, gLg] = ...
        compGrad_x(xerr, grad_x_const, beta, lambda, sens_mtrx, sparser);
      
      if do_disp
        fprintf(['%s~ '...
          'dlgr=%10.3g max_err=(%10.3g %10.3g) sqr_err=(%10.3g %10.3g)\n'],...
          prfx(), dlgr, ...
          norm(xerr.A,inf), norm(xerr.D,inf), norm(xerr.A,2), norm(xerr.D,2));
      end
      
      if out_cnt
        out_cnt = out_cnt + 1;
        out(out_cnt) = setStatus(beta, eps, iter,...
          lgrngn, sqrerr, dLd, xerr, 0);
      end
      
      if iter.I > 1 && dlgr < opts.step_ratio_thrsh * pdlgr
        break;
      end
      
    end
    
    % Update wvec
    [xerr.D, xerr.J] = sparser.optimize(xvec, lambda.D, beta.D);
    sqrerr.D = dot(xerr.D, xerr.D);
    lmderr.D = dot(lambda.D, xerr.D);
    cjerrD = sparser.compSprsVecTrnsp(xerr.D);
    lgrngn = compLgrngn(beta, [], xerr, sqrerr, lmderr);

    primal_chg = (prev_lgrngn-lgrngn)/abs(prev_lgrngn);
    if primal_chg >= eps.lgrng_chg
      [disp_db, ~] = print_psnr(prfx(), do_disp, xvec, cmp_blk_psnr, disp_db, ...
        eps.lgrng_chg, primal_chg);
      [grad_x, Agrad_x, pAgrad_x, Dgrad_x, gLg] = updateGrad_x(...
        xerr, cjerrD, grad_x_const, beta, lambda, sens_mtrx, sparser);
      continue
    end
    
    % Update lambda, Lagrangian
    % and recompute the constant part of the gradient
    prev_lgrngn = lgrngn;
    [lambda, grad_x_const] = update_lambda(beta, xerr, cjerrD,...
      lambda, grad_x_const);
    [lgrngn, sqrerr, lmderr] = compLgrngn(beta, lambda, xerr);
    
    % Check termination of outer loop
    dual_chg = (lgrngn - prev_lgrngn)/abs(lgrngn);
    chg_ok = primal_chg < eps.lgrng_chg || dual_chg < eps.lgrng_chg;

    [disp_db, disp_psnr] = print_psnr(prfx(), do_disp, xvec, cmp_blk_psnr, disp_db, ...
      eps.lgrng_chg, primal_chg, dual_chg);
    
    chg_final = chg_ok && eps.lgrng_chg == eps.lgrng_chg_final;

    if chg_ok || disp_psnr || do_disp
      % check if constraints are satisfied
      xerr0.A = comp_p(xerr.A0, xerr0.maxAerr, 0);
      xerr0.D = xerr.D;      
      cnstrnts_met = chk_constraints(eps, beta, xerr0, ...
         (disp_psnr || do_disp),  prfx);
    else
      cnstrnts_met = struct('A',false,'D',false,'all',false);
    end
    
    done = chg_final && cnstrnts_met.all;
    if done
      if out_cnt
        out_cnt = out_cnt + 1;
        out(out_cnt) = setStatus(beta, eps, iter,...
          lgrngn, sqrerr, dLd, xerr, step_size, dual_chg);
      end
      
      break;
    end
    
    % update lagrange change threshold
    if chg_ok &&  ~chg_final
        eps.lgrng_chg = max(eps.lgrng_chg_final,...
          eps.lgrng_chg * eps.lgrng_chg_rate);
    end
    
    % update beta
    [beta, beta_changed] = update_beta(beta, opts, cnstrnts_met, chg_ok);
    
    if beta_changed
      % update Lagrangian
      lgrngn = compLgrngn(beta, lambda, xerr, sqrerr, lmderr);
    end
    
    sparser.optimizeShift(xvec);
    
    % update gradient
    [grad_x, Agrad_x, pAgrad_x, Dgrad_x, gLg] = updateGrad_x(...
      xerr, cjerrD, grad_x_const, beta, lambda, sens_mtrx, sparser);

    if out_cnt
      out_cnt = out_cnt + 1;
      out(out_cnt) = setStatus(beta, eps, iter,...
        lgrngn, sqrerr, dLd, xerr, dual_chg);
    end
    
    iter.C = iter.C + 1;
    
  end
  if (opts.disp > 0 || ~isempty(opts.disp_db))
    iter.C = iter.C + iter.I;
    iter.I = 0;
    fprintf('%s solveQuant done\n', prfx());
  elseif ~done
    fprintf(...
      '%s solveExact did not converge. Cnstrnsts: %s\n  primal: %10.3g dual: %10.3g (%f)\n'...
      , prfx(), show_str(cnstrnts_met, struct(), struct('struct_marked',true)),...
      primal_chg, dual_chg, eps.lgrng_chg_final);
  end
end

function [out] = comp_p(inp, maxAerr, trans)
    out = min((inp+maxAerr), max((inp-maxAerr),0));
    if trans > 0
      indcs = find(abs(abs(inp)-maxAerr) < trans);
      x = (abs(inp(indcs)) - (maxAerr-trans))/(2*trans);
      out(indcs) = (2*trans)*(sign(inp(indcs)).*(-x.^5+1.5*x.^4));
    end
end

function [out] = comp_p_prime(inp, maxAerr, trans)
    out = double(abs(inp) > maxAerr);
    if trans > 0
      indcs = find(abs(abs(inp)-maxAerr) < trans);
      x = (abs(inp(indcs)) - (maxAerr-trans))/(2*trans);
      out(indcs) = -5*x.^4+6*x.^3;
    end
end


function [lgrngn, xvec, xerr] = ...
        move_step(xvec, xerr, drct, Adrct, Ddrct, alpha,  ...
        lambda, beta)
% function [lgrngn, xvec, xerr] = ...
%         move_step(xvec, xerr, drct, Adrct, alpha,  ...
%         sparser, lambda, beta)

    xvec = xvec + alpha*drct;
    xerr.A0 = xerr.A0 + alpha*Adrct;
    xerr.A = comp_p(xerr.A0, xerr.maxAerr, xerr.wdthAerr);
    xerr.D = xerr.D + alpha*Ddrct;
    lgrngn = compLgrngn(beta, lambda, xerr);
end

function [lambda, grad_x_const] = update_lambda(beta, xerr, cjerrD,...
        lambda, grad_x_const)
    lambda.A = lambda.A + beta.A * xerr.A;
    lambda.D = lambda.D + beta.D * xerr.D;
    grad_x_const = grad_x_const + beta.D*cjerrD;
end

function [grad_x, Agrad_x, pAgrad_x, Dgrad_x, cjerrD, gLg] = ...
    compGrad_x(xerr, grad_x_const, beta, lambda, sens_mtrx, sparser)

    cjerrD = sparser.compSprsVecTrnsp(xerr.D);
    
    [grad_x, Agrad_x, pAgrad_x, Dgrad_x, gLg] = updateGrad_x(...
        xerr, cjerrD, grad_x_const, beta, lambda, sens_mtrx, sparser);    
end

function [grad_x, Agrad_x, pAgrad_x, Dgrad_x, gLg] = updateGrad_x(...
    xerr, cjerrD, grad_x_const, beta, lambda, sens_mtrx, sparser)
    x1 = (lambda.A + beta.A*xerr.A);
    x1 = x1 .* comp_p_prime(xerr.A0, xerr.maxAerr, xerr.wdthAerr);
    grad_x = grad_x_const + beta.D*cjerrD + sens_mtrx.multTrnspVec(x1);
    Agrad_x = sens_mtrx.multVec(grad_x);
    pAgrad_x = Agrad_x;
    Dgrad_x = sparser.compSprsVec(grad_x);
    gLg = beta.D * dot(Dgrad_x, Dgrad_x) + beta.A*dot(pAgrad_x,pAgrad_x);
end

% Make a list of distances at which the Hessian
% the changes along search line
function ee = find_hessian_changes(xerr, Adrct, max_step)
    nz_indcs = find(abs(Adrct) > 1e-20);
    Ad = Adrct(nz_indcs);
    xe = xerr.A0(nz_indcs);
    ee = [(-xe-xerr.maxAerr)./Ad; (-xe+xerr.maxAerr)./Ad];
                    
    %  - Consider only the range (0,max_step) and sort
    %    uniquely in ascending order
    ee = ee(and(ee>0, ee<=max_step));
    ee = unique(ee);
end

function status = setStatus(beta, eps, iter, lgrngn, dLd, xerr, rel_chg)
    if ~isempty(xerr)
        max_A_err = max([max(xerr.N,0); max(xerr.P,0)]);
        max_D_err = max(abs(xerr.D));
    else
        max_A_err = 0;
        max_D_err = 0;
    end
    
    [st,~] = dbstack();
    status = struct(...
        'line', st(2).line,...   % line number in calling function
        'beta', beta,...
        'eps', eps,...
        'iter', iter,...
        'lgrngn', lgrngn,...
        'sqr_dLd', sqrt(dLd),...
        'max_A_err', max_A_err,...
        'max_D_err', max_D_err,...
        'rel_chg', rel_chg...
        );
    
    % disp(status);
end
