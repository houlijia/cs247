% solveExact solves the Compressive sensing reconstruction problem.
%
% Goal: Find a raw data vector x, s.t. Ax=b where b is a measurements
% vector, order(b) < order(x).  We are given a transform w=D(x) which
% sparsifies x (the order of w may be different from the order of x,
% higher or lower).  Using the sparsity we attempt to find x (or an
% approximation to it) by minimizing a target function J(w) (e.g. an L_1
% norm of w).:
%
%      min_{x,w} J(w) s.t. Dx=w, Ax=b
%
% Note that J and D are provided to this function only implicitly as
% methods of the input argument object "sparser".
%
% This function finds x by minimizing the corresponding augmented
% Lagrangian function:
%
% min_{x,w} l(x,w) = J(w) + lambda_D'(Dx-w) + (beta_D/2)||Dx-w||_2^2
%                 + lambda_A'(Ax-b) + (beta_A/2)||Ax-b||_2^2
%                 s.t. Dx=w Ax=b
%
% where lambda_D, lambda_A are Lagrange multipliers. beta_A, beta_D,
% are penalty costs which are initialized to beta_initial_A, beta_inital_D,
% respectively.
%
% The solution follows the standard augmented Lagrangian approach
% where in each iteration we find the minimum of l(x,w) given a particular
% set of Lagrange multipliers and then update the multipliers using the
% equations:
%    lambda_A <- lambda_A + beta_A(Ax-b)
%    lambda_D <- lambda_D + beta_D(Dx-w)
% This constitutes the external loop.  During the iterations the values of
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
%    g_x(x) = grad_x l(x,w) =  D'lambda_D + A'lambda_A
%        + beta_D D'(Dx-w) + beta_A A'(Ax-b)
%
% Note that the first two summands are constant w.r.t. x.  After updating
% the lagrarange multipliers this constant part can be easily updated
% because:
%
%   D'lambda_D_new + A'lambda_A_new = 
%     D'(lambda_D_old + beta_D(Dx-w)) + A'(lambda_A_old + beta_A(Ax-b)) =
%     D'lambda_D_old + A'lambda_A_old + beta_D D'(Dx-w) + beta_A A'(Ax-b)= 
%          grad_x_old l(x,w)
%
% Thus the constant part of the gradient after updating the Lagrange 
% multipliers is is just the gradient before the updating.
% Similarly, when the betas are updated all that is necessary in order
% to update gradinet is to update the variable part by appropriate scaling.
%
% Let
%
%       L = Hessian_x l(x,w) = beta_D D'D + beta_A A'A
%
%  then
%      h(d,e) = d'Le = beta_D dot(D d,D e) + beta_A dot(A d, A e)
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
%
% Inputs:
%       sens_mtrx: A SensingMatrix object.
%       b        :  input vector representing the compressed
%                   observation of a grayscale video
%       sparser  :  A BaseSparser class object.
%       opts     :  A CS_DecParams object
%       q_step   :  Quantization step of the quantizer (0=no quantization).
%       n_pix    :  Standard deviation of the noise introduced into the 
%                   measurements by the fact that pixels are uniformly
%                   quantized. This noise is normal zero mean.
%
% A note about beta and sens_mtrx: multiplication by the matrix and its
% transpose are preformed by method of the object sens_mtrx.  Somtimes
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
%            + lambda.D` xerr.D + lambda.A' xerr.A 
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
    solveExact(sens_mtrx, b, sparser, opts, q_step, msrs_pxl_stdv_err, proc_params)

  if nargin < 7
    proc_params = struct();
  end
  [prefix, cmp_blk_psnr] = parse_solver_proc_params(proc_params);
  
  b = double(b);
  
  % initialize xvec (x), beta, eps
  eps = init_solver_eps(opts, sparser, b, ...
    q_step/2, msrs_pxl_stdv_err);
  beta = init_solver_beta(opts, sens_mtrx, sparser);
  [xvec, wvec_ref] = init_solver_xvec(opts, sparser, sens_mtrx, b);
  lambda = struct('D',zeros(sparser.dimSprsVec(),1),...
    'A',zeros(length(b),1));
  disp_db = opts.disp_db;
  
  grad_x_const = struct(...
    'A',zeros(sparser.n_sigvec,1),...
    'D',zeros(sparser.n_sigvec,1),...
    'sum',zeros(sparser.n_sigvec,1)...
    );
  
  xerr = struct('A', sens_mtrx.multVec(xvec) -b, 'D', [], 'J', 0);
  [xerr.D, xerr.J] = optimize_solver_w(xvec, sparser, ...
    beta.D, lambda.D, wvec_ref);
  [lgrngn, sqrerr, lmderr] = compLgrngn(beta, lambda, xerr);
  
  iter = struct(...
    'E',0,...  % External iterations count
    'I',0, ...  % Internal iterations count in this external iteration
    'C',0 ... % Internal iterations count in previous iterations
    );
  
  if opts.cmp_solve_stats
    out_cnt = 1;
    out = setStatus(beta, eps, iter, lgrngn, sqrerr, 0, 0, xerr, 0, 0);
  else
    out_cnt = 0;
    out = [];
  end
  
  function str = prfx()
    str = sprintf('%s%d(%d:%d) ', prefix, iter.E, iter.I,iter.C);
  end
  
  done = false;
  [grad_x, Pgrad_x, cjerr, gLg] = ...
    compGrad_x(xerr, grad_x_const, beta, sens_mtrx, sparser);
  
  for ext_iter = 1:opts.max_out_iters
    if(iter.C >= opts.max_iters)
      break;
    end
    iter.E = ext_iter;
    iter.C = iter.C + iter.I;
    iter.I = 0;
    
    do_disp = opts.disp>0 && (mod(iter.E,opts.disp) == 0);
    if do_disp
      fprintf('%sSolveExact Starting iteration\n', prfx());
    end
    prev_lgrngn = lgrngn;
    
    % Run internal iterations - optimize x and w given lambda and beta
    init_cg = true;
    for int_iter = 1:min(opts.max_int_iters, opts.max_iters-iter.C)
      iter.I = int_iter;
      
      if ~init_cg
        gLd = beta.D * dot(Pgrad_x.D, Pdrct.D) + ...
          beta.scldA*dot(Pgrad_x.A,Pdrct.A);
        past_wgt = gLd/dLd;
        dLd = gLg  + past_wgt*(-2*gLd + past_wgt*dLd);
        
        % Update the direction drct based on the new gradient
        drct = - grad_x + past_wgt*drct;
        Pdrct.A = -Pgrad_x.A + past_wgt*Pdrct.A;
        Pdrct.D = -Pgrad_x.D + past_wgt*Pdrct.D;
      else
        dLd = gLg;
        drct = -grad_x;
        Pdrct = struct('A', -Pgrad_x.A, 'D', -Pgrad_x.D);
        init_cg = false;
      end
      
      step_size = - dot(drct, grad_x)/dLd;
      
      if iter.I > 1 && abs(step_size) <= opts.step_ratio_thrsh * prev_step_size
        if out_cnt
          out_cnt = out_cnt + 1;
          out(out_cnt) = setStatus(beta, eps, iter.C+iter.I,...
            lgrngn, sqrerr, dLd, xerr, step_size, 0);
        end
        if do_disp
          fprintf('%s~~ step_size=%g prev=%g DONE \n', prfx(),...
            step_size, prev_step_size);
        end
        
        iter.I = iter.I-1;
        break;
      end
      prev_step_size = step_size;
      
      step = comp_step(drct, Pdrct, xerr, lambda);
      [lgrngn, xvec, xerr, sqrerr, lmderr] =...
        move_step(xvec, xerr, step, step_size, sqrerr, lmderr,...
        sparser, lambda, beta, wvec_ref);
            
      % Update gradient
      [grad_x, Pgrad_x, cjerr, gLg] = ...
        compGrad_x(xerr, grad_x_const, beta, sens_mtrx, sparser);
      
      if do_disp
        fprintf(['%s~ J_opt=%12.05g '...
          'step=%6.4f max_err=(%10.3g %10.3g) sqr_err=(%10.3g %10.3g)\n'],...
          prfx(),xerr.J, step_size, norm(xerr.A,inf), norm(xerr.D,inf),...
          norm(xerr.A,2), norm(xerr.D,2));
      end
      
      if out_cnt
        out_cnt = out_cnt + 1;
        out(out_cnt) = setStatus(beta, eps, iter,...
          lgrngn, sqrerr, dLd, xerr, step_size, 0);
      end
      
    end
    
    primal_chg = (prev_lgrngn-lgrngn)/abs(prev_lgrngn);
    
    % Update lambda, Lagrangian
    % and recompute the constant part of the gradient
    prev_lgrngn = lgrngn;
    [lambda, lmderr, grad_x_const] = update_lambda(beta, xerr, cjerr,...
      lambda, grad_x_const, grad_x);
    lgrngn = compLgrngn(beta, lambda, xerr, lmderr, sqrerr);
    
    % Check termination of outer loop
    dual_chg = (lgrngn - prev_lgrngn)/abs(lgrngn);    
    chg_ok = primal_chg < eps.lgrng_chg && dual_chg < eps.lgrng_chg;
    
    [disp_psnr, psnr, disp_db] = chk_disp_db(xvec, cmp_blk_psnr,  disp_db);
    if disp_psnr || do_disp
      if chg_ok
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
      fprintf('%s lgrng_chg: prml=%10.3g dual=%10.3g (%g) %s\n', prfx(),...
        primal_chg, dual_chg, eps.lgrng_chg, ok_str);
    end
    chg_final = chg_ok && eps.lgrng_chg == eps.lgrng_chg_final;
    
    if chg_ok || disp_psnr || do_disp 
      % check if constraints are satisfied
       cnstrnts_met = chk_constraints(eps, beta, xerr, ...
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
    
    [beta, beta_changed] = update_beta(beta, opts, cnstrnts_met, chg_ok);
    
    % update gradient
    [grad_x, Pgrad_x, gLg] = updateGrad_x(...
      cjerr, grad_x_const, beta, sens_mtrx, sparser);
    
    if beta_changed.A || beta_changed.D
      % update Lagrangian
      lgrngn = compLgrngn(beta, lambda, xerr, lmderr, sqrerr);
    end
    
    if out_cnt
      out_cnt = out_cnt + 1;
      out(out_cnt) = setStatus(beta, eps, iter,...
        lgrngn, sqrerr, dLd, xerr, step_size, dual_chg);
    end
    
  end
  if (opts.disp > 0  || ~isempty(opts.disp_db))
    iter.C = iter.C + iter.I;
    iter.I = 0;
    fprintf('%s solveExact done\n', prfx());
  end
  
end

function [grad_x, Pgrad_x, cjerr, gLg] = ...
    compGrad_x(xerr, grad_x_const, beta, sens_mtrx, sparser)
  
  cjerr = struct('A', sens_mtrx.multTrnspVec(xerr.A),...
    'D', sparser.compSprsVecTrnsp(xerr.D));
  
  [grad_x, Pgrad_x, gLg] = updateGrad_x(...
    cjerr, grad_x_const, beta, sens_mtrx, sparser);
end

function [grad_x, Pgrad_x, gLg] = updateGrad_x(...
    cjerr, grad_x_const, beta, sens_mtrx, sparser)
  grad_x = ...
    beta.A*cjerr.A + beta.D*cjerr.D + grad_x_const.sum;
  Pgrad_x = struct('A', sens_mtrx.multVec(grad_x),...
    'D', sparser.compSprsVec(grad_x));
  gLg = beta.D * dot(Pgrad_x.D, Pgrad_x.D) + ...
    beta.scldA*dot(Pgrad_x.A,Pgrad_x.A);
end

function [lambda, lmderr, grad_x_const] = update_lambda(beta, xerr, cjerr,...
    lambda, grad_x_const, grad_x)
  lambda.A = lambda.A + beta.scldA * xerr.A;
  lambda.D = lambda.D + beta.D * xerr.D;
  lmderr = struct('A', dot(lambda.A, xerr.A),...
    'D', dot(lambda.D, xerr.D));
  grad_x_const.A = grad_x_const.A + beta.A*cjerr.A;
  grad_x_const.D = grad_x_const.D + beta.D*cjerr.D;
  grad_x_const.sum = grad_x;
  
end

function step = comp_step(drct, Pdrct, xerr, lambda)
  step = struct('X', drct, 'A', Pdrct.A, 'D', Pdrct.D);
  step.LA = dot(lambda.A, step.A);
  step.LD = dot(lambda.D, step.D);
  step.EA = dot(xerr.A, step.A);
  step.ED = dot(xerr.D, step.D);
  step.A2 = dot(step.A, step.A);
  step.D2 = dot(step.D, step.D);
end

function [lgrngn, xvec, xerr, sqrerr, lmderr] = ...
    move_step(xvec, xerr, step, alpha, sqrerr, lmderr,...
    sparser, lambda, beta, wvec_ref)
  
  xvec = xvec + alpha*step.X;
  xerr.A = xerr.A + alpha*step.A;
  [xerr.D, xerr.J] = optimize_solver_w(xvec, sparser, ...
    beta.D, lambda.D, wvec_ref);
  lmderr.A =  lmderr.A + alpha*step.LA;
  lmderr.D = dot(lambda.D,xerr.D);
  sqrerr.A = sqrerr.A + 2*alpha*step.EA + alpha*alpha*step.A2;
  sqrerr.D = dot(xerr.D,xerr.D);
  lgrngn = compLgrngn(beta, [], xerr, lmderr, sqrerr);
end

function status = setStatus(beta, eps, iter, lgrngn, sqrerr, dLd, xerr, ...
    step_size, dual_chg)
  if ~isempty(xerr)
    max_A_err = max(abs(xerr.A));
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
    'sqrerr', sqrerr,...
    'sqr_dLd', sqrt(dLd),...
    'max_A_err', max_A_err,...
    'max_D_err', max_D_err,...
    'step_size', step_size,...
    'dual_chg', dual_chg...
    );
  
  % disp(status);
end


