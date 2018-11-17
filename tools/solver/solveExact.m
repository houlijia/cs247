% solveExact solves the Compressive sensing reconstruction problem.
%
% Goal: Find a raw data vector x, s.t. Ax=b where b is a measurements
% vector, order(b) < order(x).  We are given a transform w=D(x) which
% sparsifies x (the order of w may be different from the order of x,
% higher or lower).  Using the sparsity we attempt to find x (or an
% approximation to it) by minimizing a target function J(w) (e.g. an L_1
% norm of w).:
%
%      min_{x,v,w} J(w) s.t. Dx=w, v=x, Ax=b
%
% Note that J and D are provided to this function only implicitly as
% methods of the input argument object "sparser".
%
% This function finds x by minimizing the corresponding augmented
% Lagrangian function:
%
% min_{x,w} l(x,v,w) = J(w) + lambda_D'(Dv-w) + (beta_D/2)||Dv-w||_2^2
%                 + lambda_V'(x-v) + (beta_V/2)||x-v||_2^2
%                 + lambda_A'(Ax-b) + (beta_A/2)||Ax-b||_2^2
%                 s.t. Dx=w Ax=b
%
% where lambda_D, lambda_A, lambda_V are Lagrange multipliers. beta_A, beta_D,
% beta_V are penalty costs which are initialized to beta_initial_A, beta_inital_D,
% respectively.
%
% The solution follows the standard augmented Lagrangian approach
% where in each iteration we find the minimum of l(x,w) given a particular
% set of Lagrange multipliers and then update the multipliers using the
% equations:
%    lambda_A <- lambda_A + beta_A(Ax-b)
%    lambda_D <- lambda_D + beta_D(Dv-w)
%    lambda_V <- lambda_V + beta_V(x-v)
% This constitutes the external loop.  During the iterations the values of
% beta_A, beta_D, beta_V are increased until reaching a final value:
%
% The calculation of x, v and w is done
% in an ineternal loop which alternatiely minimizes l(x,v,w) with respect to
% x, v, and w.  
%
% The sensing matrix and the sparser are given as input arguments and are
% expected to be objects with methods. The minimization of w
% given v is done by a method of the sparser and is
% not a part of this function description. The minimization of x given v,w
% and of v given x,w are done by running a few steps of congjugate gradient.
% This is done using the equations for the gradient and the Hessian of the 
% Lagrangian:
%
%    g_x(x,v,w) = grad_x l(x,v,w) =  A'lambda_A + lambda_V
%        + beta_A A'(Ax-b) + beta_V(x-v)
%    g_v(x,v,w) = grad_v l(x,v,w) =  D'lambda_D - lambda_V
%        + beta_D D'(Dv-w) - beta_V(x-v)
%
% Note that the first two summands are constant w.r.t. x.  After updating
% the lagrarange multipliers this constant part can be easily updated
% because:
%
%   A'lambda_A_new + lambda_V = 
%     A'(lambda_A_old + beta_A(Ax-b)) +(lambda_V_old + beta_V(x-v)) =
%     A'lambda_A_old + lambda_V_old + beta_A A'(Ax-b) + beta_V(x-v)= 
%          grad_x_old l(x,v,w)
%   D'lambda_D_new - lambda_V_new = 
%     D'(lambda_D_old + beta_D(Dv-w)) - (lambda_V_old + beta_V(x-v)) =
%     D'lambda_D_old - lambda_V_old + beta_D D'(Dv-w) - beta_V (v-w)= 
%          grad_v_old l(x,w)
%
% Thus the constant part of the gradients after updating the Lagrange 
% multipliers is just the gradients before the updating.
% Similarly, when the betas are updated all that is necessary in order
% to update gradinet is to update the variable part by appropriate scaling.
%
% Let
%
%       L_x = Hessian_x l(x,w) = beta_v I + beta_A A'A
%       L_x = Hessian_x l(x,w) = beta_v I + beta_D D'D
%
%  then
%      h_x(d,e) = d'L_x e = beta_V dot(d,e) + beta_A dot(A d, A e)
%      h_v(d,e) = d'L_v e = beta_V dot(d,e) + beta_D dot(D d, D e)
%
%  The conjugate gradient for x starts with an inital guess x as
%  follows:
%    Initialize: 
%      g = g_x(x),
%      Compute A'g and h_x(g,g)
%      set d = - g, A'd = -A'g, h(d,d) = h(g,g)
%    Repeat n_A times:
%         s <- ((g'd)/h_x(d,d))   << compute step size
%         if ||s|| < threshold break << check for convergence 
%         x  <-  x + s*d  << Update x
%         g  <- g + (s*beta_A)*(A'*A*d) + (s*beta_V)*d << new grad_x
%         d  <- -g + (h_x(g,d)/h_x(d,d))d  << update search direction
%
%  Similarly, the conjugate gradient for v starts with an inital guess v as
%  follows:
%    Initialize: 
%      g = g_x(x),
%      Compute D'g and h_v(g,g)
%      set d = - g, D'd = -D'g, h(d,d) = h(g,g)
%    Repeat n_A times:
%         s <- ((g'd)/h_v(d,d))   << compute step size
%         if ||s|| < threshold break << check for convergence 
%         v  <-  v + s*d  << Update v
%         g  <- g + (s*beta_D)*(D'*D*d) + (s*beta_V)*d << new grad_v
%         d  <- -g + (h_v(g,d)/h_v(d,d))d  << update search direction
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
% xerr.DV = DV-w
% xerr.V = x-v
% xerr.J = J(w)
% sqrerr.A = ||Ax -b ||^2
% sqrerr.DV = ||Dv-w||^2
% sqrerr.D  = ||Dx-w||^2
% sqrerr.V = ||x-v||^2
% cnj_err.A  = A'(Ax-b)
% cnj_err.DV = D'(Dv-w) (coefficients beta and mu are not included)
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
    solveExact(sens_mtrx, b, sparser, opts, q_step, msrs_pxl_stdv_err, ...
    proc_params)

  if nargin < 7
    proc_params = struct();
  end
  [prefix, cmp_blk_psnr] = parse_solver_proc_params(proc_params);
  disp_db = opts.disp_db;

  function str = prfx()
    str = sprintf('%s%d(%d:%d) ', prefix, iter.E, iter.I,iter.C);
  end
  
%   use_gpu = isa(b,'gpuArray');
  
  sparser.reset();
  
  % initialize xvec (x), beta, eps
  eps = init_solver_eps(opts, sparser, b, ...
    q_step/2, msrs_pxl_stdv_err);
  beta = init_solver_beta(opts, sens_mtrx, sparser);
  xvec = init_solver_xvec(opts, sens_mtrx, b);
  lambda = struct('A',sens_mtrx.zeros(length(b),1), 'D', ...
    sens_mtrx.zeros(sparser.n_sprsvec,1));


  xerr = struct(...
    'A', sens_mtrx.multVec(xvec) -b, ... A*x-b
    'D', [], ... Unnormalized sparser error Dx-w
    'J', [] ... ||w||_1
    );

  
  [xerr.D, xerr.J] = sparser.optimize(xvec, lambda.D, beta.D);
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
  
%   function chk_xerr()
%     xer = struct(...
%       'A', sens_mtrx.multVec(xvec) - b,...
%       'D', sparser.compSprsVec(xvec) - wvec);
%     errs = [norm(xerr.B-xer.B,inf) norm(xerr.D-xer.D,inf)];
%     if sum(errs) > 1e-8
%       warning('errs too large');
%     end
%   end
  
%   function grd = calc_grad()
%     lgrng = compLgrngn(beta, lambda, xerr);
%     dlta = 0.001;
%     xv = xvec;
%     xer = xerr;
%     grd = grad;
%     for k=1:length(xerr.B)
%       xv(k) = xv(k) + dlta;
%       xer.B = sens_mtrx.multVec(xv) - b;
%       xer.D = sparser.compSprsVec(xv) - wvec;
%       lgrn = compLgrngn(beta, lambda, xer);
%       grd(k) = (lgrn-lgrng)/dlta;
%       xv(k) = xvec(k);
%     end
%   end
  
  % External loop
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
      fprintf('%sSolveExact Starting iteration\n', prfx());
    end
    prev_lgrngn = lgrngn;
    
    % Run internal iterations - optimize x given v, w, lambda and beta
    init_cg = true;
    
%     chk_xerr();
        
    for int_iter = 1:min(opts.max_int_iters, opts.max_iters-iter.C)
      iter.I = int_iter;
      
      % compute gradient
      [grad, Pgrad, gLg] = compGrad(xerr, beta, lambda, sens_mtrx, sparser);
      
      if ~init_cg
        gLd = hessProd(beta, Pgrad, Pdrct);
        past_wgt = gLd/dLd;
        dLd = gLg  + past_wgt*(-2*gLd + past_wgt*dLd);
        
        % Update the direction drct based on the new gradient
        drct = -grad + past_wgt*drct;
        Pdrct = struct(...
          'A', -Pgrad.A + past_wgt*Pdrct.A, ...
          'D', -Pgrad.D + past_wgt*Pdrct.D);
      else
        dLd = gLg;
        drct = -grad;
        Pdrct = struct('A', -Pgrad.A, 'D', -Pgrad.D);
        init_cg = false;
      end
      
      step_size = -dot(drct, grad)/dLd;
      
      lgr = lgrngn; % last inner iteration lagrangian
      if iter.I > 1
        pdlgr = dlgr;
      end
      
      [xvec, xerr] =  move_step(xvec, xerr, drct, Pdrct, step_size);
%       chk_xerr();
      
      % Update wvec
      [xerr.D, xerr.J] = sparser.optimize(xvec, lambda.D, beta.D);

      % Compute lagrangian and check convergence
      [lgrngn, sqrerr, lmderr] = compLgrngn(beta, lambda, xerr);
      dlgr = lgr - lgrngn;
      
      if do_disp
        fprintf(['%s~ '...
          'Stp=%g dlgr=%10.3g max_err=(%10.3g %10.3g) L2_err=(%10.3g %10.3g)\n'],...
          prfx(), step_size, dlgr, norm(xerr.A,inf), norm(xerr.D,inf),...
          norm(xerr.A,2), norm(xerr.D,2));
      end
      
      if out_cnt
        out_cnt = out_cnt + 1;
        out(out_cnt) = setStatus(beta, eps, iter,...
          lgrngn, sqrerr, dLd, xerr, step_size, 0);
      end
      
      if int_iter > 1 && dlgr < opts.step_ratio_thrsh * pdlgr
        break;
      end
      
    end
    
    % Check primal change
    primal_chg = (prev_lgrngn-lgrngn)/abs(prev_lgrngn);
    if primal_chg >= eps.lgrng_chg
      [disp_db, ~] = print_psnr(prfx(), do_disp, xvec, cmp_blk_psnr, disp_db, ...
        eps.lgrng_chg, primal_chg);
      continue
    end
    
    % Update lambda and Lagrangian
    prev_lgrngn = lgrngn;
    [lambda, lmderr] = update_lambda(beta, xerr, sqrerr, lambda, lmderr);
    lgrngn = compLgrngn(beta, lambda, xerr, sqrerr, lmderr);
    
    % Check termination of outer loop
    dual_chg = (lgrngn - prev_lgrngn)/abs(lgrngn);    
    chg_ok = primal_chg < eps.lgrng_chg && dual_chg < eps.lgrng_chg;
    
    [disp_db, disp_psnr] = print_psnr(prfx(), do_disp, xvec, cmp_blk_psnr,...
      disp_db, eps.lgrng_chg, primal_chg, dual_chg);
    
    if chg_ok
      shft_done = sparser.optimizeShift(xvec);
      if shft_done == 1;
        lamda.D = zeros(size(lambda.D));
        lamda.A = zeros(size(lambda.A));
        xvec = zeros(size(xvec));
        xerr.A = -b;
        [xerr.D, xerr.J] = sparser.optimize(xvec, lambda.D, beta.D);
        [lgrngn, sqrerr, lmderr] = compLgrngn(beta, lambda, xerr);
%         dual_chg = (lgrngn - prev_lgrngn)/abs(lgrngn);
        chg_ok = false;
      end
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
        
    if beta_changed
      % update Lagrangian
      lgrngn = compLgrngn(beta, lambda, xerr, sqrerr, lmderr);
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
  elseif ~done
    fprintf(...
      '%s solveExact did not converge. primal: %10.3g dual: %10.3g (%f)\n'...
      , prfx(), primal_chg, dual_chg, eps.lgrng_chg_final);
    chk_constraints(eps, beta, xerr, true,  prfx);  
  end
  
%   if use_gpu
%     xvec = gather(xvec);
%     lambda.A = gather(lambda.A);
%     lambda.D = gather(lambda.D);
%   end
  
end

function [grad, Pgrad, gLg] = compGrad(xerr, beta, lambda, ...
    sens_mtrx, sparser)
  cjerr = struct(...
    'A', sens_mtrx.multTrnspVec(lambda.A + beta.A*xerr.A),...
    'D', sparser.compSprsVecTrnsp(lambda.D + beta.D*xerr.D));
   grad = cjerr.A + cjerr.D;
  
  Pgrad = struct(...
    'A', sens_mtrx.multVec(grad),...
    'D', sparser.compSprsVec(grad));
  
  gLg = hessProd(beta, Pgrad);
end

function val = hessProd(beta, Pf, Pg)
  if nargin < 3
    Pg = Pf;
  end
 val = beta.A*dot(Pf.A,Pg.A) + beta.D*dot(Pf.D,Pg.D);
end

function [xvec, xerr] = move_step(xvec, xerr, drct, Pdrct, alpha)
  xvec = xvec + alpha*drct;
  xerr.A = xerr.A + alpha*Pdrct.A;
  xerr.D = xerr.D + alpha*Pdrct.D;
end

function [lambda, lmderr] = update_lambda(beta, xerr, sqrerr, lambda, lmderr)
  lambda.A = lambda.A + beta.A * xerr.A;
  lambda.D = lambda.D + beta.D * xerr.D;
  lmderr.A = lmderr.A + beta.A * sqrerr.A;
  lmderr.D = lmderr.D + beta.D * sqrerr.D;
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




