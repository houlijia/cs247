%
% Goal: solve   min sum ||D_i u||_1
%                  s.t. Au = b
%       to recover video u from encoded b,
%       which is equivalent to solve       min sum ||w_i||
%                                          s.t. D_i u = w_iW
%                                               Au = b
% Here we use 2D anisotropic total variation as objective function.
%
% TVAL3 solves the corresponding augmented Lagrangian function:
%
% min_{u,w} sum ( ||w_i||_1 - sigma'(D_i u - w_i) + beta/2||D_i u - w_i||_2^2 )
%                   - delta'(Au-b) + mu/2||Au-b||_2^2 ,
%
% by an alternating algorithm:
% i)  while not converge
%     1) Fix w^k, do Gradient Descent to
%            - sigma'(Du-w^k) - delta'(Au-b) + beta/2||Du-w^k||^2 + mu/2||Au-b||^2;
%            u^k+1 is determined in the following way:
%         a) compute step length tau > 0 by BB formula
%         b) determine u^k+1 by
%                  u^k+1 = u^k - alpha*g^k,
%            where g^k = -D'sigma - A'delta + beta D'(Du^k - w^k) + mu A'(Au^k-b),
%            and alpha is determined by Amijo-like nonmonotone line search;
%     2) Given u^k+1, compute w^k+1 by shrinkage
%                 w^k+1 = shrink(Du^k+1-sigma/beta, 1/beta);
%     end
% ii) update Lagrangian multipliers by
%             sigma^k+1 = sigma^k - beta(Du^k+1 - w^k+1)
%             delta^k+1 = delta^k - mu(Au^k+1 - b).
% iii)accept current u as the initial guess to run the loop again
%
% Inputs:
%       A        : either an matrix representing the measurement or a struct
%                  with 2 function handles:
%                           A(x,1) defines @(x) A*x;
%                           A(x,2) defines @(x) A'*x;
%       b        :  input vector representing the compressed
%                   observation of a grayscale video
%       opts     :  structure to restore parameters
%
%
% variables in this code:
%
% cerr.A = Au-b
% cerr.D  Du-w
% cjerr.A = A'(Au-b)
% cjerr.D = D'(Du-b)
% sum_abs_W = sum ||wi||_1
% sqerr.D = ||Du-w||^2 (at current w).
% sqerr.A = ||Au-b||^2
% lgerr.D = sigma'(Du-w)
% lgerr.A = delta'(Au-b)
%
%   f  = sum_abs_W + beta/2 sqerr.D + mu/2 sqerr.A - lgerr.D - lgerr.A
%
% A 'p' suffix in variable names indicates previosu values.
% uup - norm(U-Up) - norm of change in U 
%
%
% Written by: Chengbo Li @ Bell Laboratories, Alcatel-Lucent
% Computational and Applied Mathematics department, Rice University
% 06/12/2010
function [U, out] = TVAL3_CVS_D2_a(sens_mtrx, sparser, b,opts)
    % get or check opts
    opts = TVAL3_CVS_opts(opts);
    
    sns_mtrx = sens_mtrx.getHandle_multLR();
    
    % mark important constants
    tol_inn = opts.tol_inn;
    tol_out = opts.tol;
    nonmonotone_wgt = opts.gam;  % nonmonotonicity weight (0=monotone)
    
    % calculate A'*b
    b = double(b);
    
    % initialize U, beta
    U = init_U(sns_mtrx, sparser, b,opts);
    beta = init_solver_beta(opts, sens_mtrx, sparser);
    rcdU = U;
    nrmrcdU = norm(rcdU(:));
    nrmb = norm(b);
    
    % initialize multiplers
    lambda = struct('D', zeros(sparser.dimSprsVec(),1),...
        'A', zeros(length(b),1));
    
    % initialize D^T sigma + A^T delta
    grad_x_const = struct(...
        'A',zeros(sparser.n_sigvec,1),...
        'D',zeros(sparser.n_sigvec,1),...
        'sum',zeros(sparser.n_sigvec,1)...
    );
    
    % initialize out.errTrue which records the true relative error
    % Razi:  If present in  opts, Ut is supposed to provide the true value of U
    % for debugging purposes
    if isfield(opts,'Ut')
        Ut = opts.Ut;        %true U, just for computing the error
        nrmUt = norm(Ut);
    else
        Ut = [];
    end
    if ~isempty(Ut)
        out.errTrue = norm(U - Ut);
    end
    
    % prepare for iterations
    out.res = [];      % record errors of inner iterations--norm(H-Hp)
    out.reer = [];     % record relative errors of outer iterations
    out.innstp = [];   % record RelChg of inner iterations
    out.itrs = [];     % record # of inner iterations
    out.itr = Inf;     % record # of total iterations
    out.f = [];        % record values of augmented Lagrangian fnc
    out.cnt = [];      % record # of back tracking
    out.sum_abs_W = []; out.sqerr = []; out.lgerr = [];
    out.tau = []; out.alpha = []; out.C = [];
    
    cjerrp = [];
    
    cerr = struct('A',0,'D',0);
    [cerr.D, sum_abs_W, W, DU] = optimize_solver_w(U, sparser, beta.D, lambda.D);
    [cerr, sqerr, lgerr, cjerr, f, Au] = get_g(U, cerr, sparser,...
        W, sum_abs_W,beta,sns_mtrx,b,lambda);
    
    gradU = comp_grad(cjerr, grad_x_const, beta);
    
    count = 1; sum_itrs = 0;
    Q = 1; C = f;                     % Q, C: costant
    out.f = [out.f; f]; out.C = [out.C; C];
    out.sum_abs_W = [out.sum_abs_W; sum_abs_W]; 
    out.sqerr = sqerr;
    out.lgerr = lgerr;
    
    for ii = 1:opts.maxit
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            fprintf('outer iter = %d, total iter = %d, \n',count,ii);
        end
        
        % compute tau, the first guess of (minus) step size. In the first
        % iteration, using the formula tau = g'g/g'Hg, where H is the
        % Hessian of the object function f.  In the first iteration g is 
        % the gradient gradU.  In subsequent iterations we replace it by
        % the differnce from the previous step, g=U-Up (uup)  (Why?).
        % replac
        if ~isempty(cjerrp)
            dcjerr = struct('A', cjerr.A-cjerrp.A, 'D', cjerr.D-cjerrp.D);
            ss = uup'*uup;                      % ss: constant
            sy = uup'*(beta.D*dcjerr.D + beta.A*dcjerr.A);       % sy: constant

            % compute BB step length
            tau = abs(ss/max(sy,eps));               % tau: constant
%            tau_ratio = tau/comp_tau(sns_mtrx, sparser, gradU,beta);

            %fst_itr = false;
        else
            % do Steepest Descent at the 1st ieration
            tau = comp_tau(sns_mtrx, sparser, gradU, beta);
%            tau_ratio = 1;
            % mark the first iteration
            %fst_itr = true;
        end
        
        % keep the previous values
        Up = U; cjerrp = cjerr; Aup = Au; DUp = DU;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ONE-STEP GRADIENT DESCENT %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        taud = tau*gradU;
        U = U - taud;
        DU = sparser.compSprsVec(U);
        
        [cerr, sqerr,lgerr,cjerr,f,Au] = get_g(U, cerr, sparser,...
            W, sum_abs_W,beta,sns_mtrx,b,lambda);
        
        % Nonmonotone Line Search
        % Unew = Up + alpha*(U - Up).  Applying Armijio criterion to adjust
        % the step size size prevent a step size which is too long. alpha is 
        % is started as 1 (step of taud) and decreased until the value of
        % f(Unew) does not exceed C - alpha*const. C represent the starting point.
        % In monotome decrease it is simply f(Up).  Otherwise it is some
        % weighted avearge of f at several previous points.  alpha*const
        % represents a required reduction in f based on linear
        % extrapolation using the gradient.  The expected reduction
        % (that is, if the Hessian was constant and the problem was
        % quadratic) is alpha*(gradU'*taud).  We multiply the expected
        % slope, (gradU'*taud) by a constant smaller than 1, to allow
        % variations due to the fact that the our problem is not really
        % quadratic. 
        %
        % During the iteration the flag d_not_updated indicates that differences
        % dAu, dDu and dcjerr are up not to date.
        alpha = 1;
        du = U - Up;
        
        const = opts.c*(gradU'*taud);
        
        cnt = 0; d_not_updated = true;
        while f > C - alpha*const
            if cnt == 5
                % As alpha is reduced, Unew gets closer to Up and if C>f(x)
                % it may be impossible to meet the crierion for any
                % small alpha. On the other hand, the fact that after so
                % many iterations we did not meet the criterion suggests
                % that indeed C>f(x).  get out of this bad place by computing
                % the step size as if from scratch, using comp_tau() 
                % without a line search, and then we reduce the amount of
                % nonmonotonicity for the next line search.
                % 
                nonmonotone_wgt = opts.rate_gam*nonmonotone_wgt;
                
                % give up and take Steepest Descent step
                if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
                    disp('    count of back tracking attains 5 ');
                end
                
                tau = comp_tau(sns_mtrx,sparser, gradU,beta);
                U = Up - tau*gradU;
                [cerr.D, sum_abs_W, W, DU] = ...
                    optimize_solver_w(U, sparser, beta.D, lambda.D);
                [cerr,sqerr,lgerr,cjerr,f,Au] = get_g(U, cerr, sparser,...
                     W, sum_abs_W,beta,sns_mtrx,b,lambda);
                alpha = 0; % remark the failure of back tracking
                break;
            end
            if d_not_updated
                dcjerr = struct('A', cjerr.A - cjerrp.A,...
                    'D', cjerr.D - cjerrp.D);
                dAu = Au - Aup;                 % dAu: m
                dDU = DU - DUp;
                d_not_updated = false;
            end
            alpha = alpha*opts.gamma;
            [U,cerr, sqerr,lgerr,cjerr,f,DU,Au] = update_g(sparser,...
                sum_abs_W,alpha,beta,Up,du,cjerrp,dcjerr,Aup,dAu,W,...
                DUp, dDU, b,lambda);
            cnt = cnt + 1;
        end
%        fprintf('alpha=%g tau ratio=%g ratio =%g\n', alpha, tau_ratio, alpha/tau_ratio);
        % if back tracking is succeceful, then recompute
        if alpha ~= 0
            [cerr.D, sum_abs_W, W, DU] = ...
                optimize_solver_w(U, sparser, beta.D, lambda.D);
            
            % update parameters related to W
            [sqerr.D,lgerr.D,cjerr.D] = update_W(cerr.D, sparser, lambda.D);
            
            % update f
            f = comp_lgrngn(beta, sum_abs_W, sqerr, lgerr);
        end
        
        % update reference value
        Qp = Q; 
        Q = nonmonotone_wgt*Qp + 1; 
        if isstruct(f)
            fprintf('f is struct\n');
        end
        C = (nonmonotone_wgt*Qp*C + f)/Q;
        uup = U - Up;            % uup: pq
        nrmuup = norm(uup);                   % nrmuup: constant
        
        out.res = [out.res; nrmuup];
        out.f = [out.f; f]; out.C = [out.C; C]; out.cnt = [out.cnt;cnt];
        out.sum_abs_W = [out.sum_abs_W; sum_abs_W]; 
        out.sqerr = [out.sqerr; sqerr]; 
        out.lgerr = [out.lgerr; lgerr];
        out.tau = [out.tau; tau]; out.alpha = [out.alpha; alpha];
        
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            fprintf('       ||D(U)-W|| = %5.3f, ||Au-f||/||f|| = %5.3f, ',...
                sqrt(sqerr.D), sqrt(sqerr.A)/nrmb);
        end
        
        if ~isempty(Ut)
            errT = norm(U - Ut(:));
            out.errTrue = [out.errTrue; errT];
            if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
                fprintf('  ||Utrue-U||(/||Utrue||) = %5.3f(%5.3f%%), ',errT, 100*errT/nrmUt);
            end
        end
        
        gradU = comp_grad(cjerr, grad_x_const, beta);
        
        % compute relative change or optimality gap
        if opts.StpCr == 1          % relative change
            nrmup = norm(Up);
            RelChg = nrmuup/nrmup;
            if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
                fprintf('    ||Uprvs-U||/||Uprvs|| = %5.3f; \n', RelChg);
            end
        else                       % optimality gap
            RelChg = norm(gradU) / beta.D;
            if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
                fprintf('    optimality gap = %5.3f; \n', RelChg);
            end
        end
        out.innstp = [out.innstp; RelChg];
        
        if (RelChg < tol_inn || ii-sum_itrs >= opts.maxin)% && ~fst_itr
            count = count + 1;
            RelChgOut = norm(U(:)-rcdU(:))/nrmrcdU;
            out.reer = [out.reer; RelChgOut];
            rcdU = U;
            nrmrcdU = norm(rcdU(:));
            if isempty(out.itrs)
                out.itrs = ii;
            else
                out.itrs = [out.itrs; ii - sum(out.itrs)];
            end
            sum_itrs = sum(out.itrs);
            
            % stop if already reached final multipliers
            if ( RelChgOut < tol_out && norm(cerr.A,inf) < 1 &&...
                    norm(cerr.D,inf) <0.1 ) || count > opts.maxcnt
                break;
            end
            
            % update multipliers
            % Razi: f is changed as a result of this, but it is anyway recomputed
            % below.
            [lambda, lgerr, grad_x_const] = update_mlp(beta, cerr, cjerr,...
                lambda, grad_x_const, gradU);

            % update penality parameters for continuation scheme
            beta = update_beta(beta, opts, grad_x_const, cerr);
            
            % update function value, gradient, and relavent constant
            f = comp_lgrngn(beta, sum_abs_W, sqerr, lgerr);
            
            gradU = comp_grad(cjerr, grad_x_const, beta);
            
            %initialize the constants
            cjerrp = [];
            nonmonotone_wgt = opts.gam; Q = 1; C = f;
        end
        
    end
    
    out.itr = ii;
    if opts.isreal
        U = real(U);
    end
    % fprintf('Attain the maximum of iterations %d. \n',opts.maxit);
end

% Initialize U
function U = init_U(sns_mtrx, sparser, b, opts)
    if isscalar(opts.init)
        switch opts.init
            case 0, U = zeros(sparser.n_sigvec,1);
            case 1, U = sns_mtrx(b,true);
        end
    else
        U = opts.init;
    end
end

% This function expects to find cerr.D as input and updates cerr.A on
% output.
function [cerr, sqerr,lgerr,cjerr,f,Au] = get_g(U, cerr, sparser,...
    W, sum_abs_W, beta,sns_mtrx,b,lambda)

    sqerr = struct('A',0,'D',0);
    lgerr = struct('A',0,'D',0);
    cjerr = struct('A',0,'D',0);
    
    cerr.D = sparser.compSprsVec(U) - W;
    [sqerr.D, lgerr.D, cjerr.D] = update_W(cerr.D, sparser, lambda.D);
    
    Au = sns_mtrx(U,false);
    cerr.A = Au-b;
    cjerr.A = sns_mtrx(cerr.A,true);
    sqerr.A = dot(cerr.A, cerr.A);
    lgerr.A = dot(lambda.A,cerr.A);
    
    f = comp_lgrngn(beta, sum_abs_W, sqerr, lgerr);
end

% Compute the value of the Lagrangian
function f = comp_lgrngn(beta, sum_abs_W, sqerr, lgerr)
    f = sum_abs_W + beta.D/2*sqerr.D + beta.A/2*sqerr.A + lgerr.D + lgerr.A;
end

function gradU = comp_grad(cjerr, grad_x_const, beta)
    % compute gradient
    gradU = beta.D*cjerr.D + beta.A*cjerr.A + grad_x_const.sum;
end    

function tau = comp_tau(sns_mtrx,sparser,gradU, beta)
    Dd = sparser.compSprsVec(gradU);
    Ad = sns_mtrx(gradU,false);                        %Ad: m
    dLd = beta.D*dot(Dd,Dd)+ beta.A*dot(Ad,Ad);         % dDd: cosntant
    % compute Steepest Descent step length
    tau = abs((gradU'*gradU)/dLd);
end

function [U,cerr, sqerr,lgerr,cjerr,f,DU,Au] = update_g(sparser,...
    sum_abs_W,alpha,beta,Up,du,cjerrp,dcjerr,Aup,dAu,W,DUp, dDU,...
    b,lambda)
          
    cjerr = struct('A',0,'D',0);
    cjerr.A = cjerrp.A + alpha*dcjerr.A;
    cjerr.D = cjerrp.D + alpha*dcjerr.D;
    U = Up + alpha*du;
    Au = Aup + alpha*dAu;
    DU = DUp + alpha*dDU;
    
    sqerr = struct('A',0,'D',0);
    lgerr = struct('A',0,'D',0);
    cerr = struct('A',0,'D',0);
    cerr.A = Au-b;
    cerr.D = DU-W;
    [sqerr.D, lgerr.D] = update_W(cerr.D, sparser, lambda.D);
    sqerr.A = dot(cerr.A,cerr.A);
    lgerr.A = dot(lambda.A,cerr.A);
    f = comp_lgrngn(beta, sum_abs_W, sqerr, lgerr);
end

function [sqerr_D,lgerr_D, cjerr_D] = update_W(cerr_D, sparser, sigma)
    % update parameters because W was updated
    sqerr_D = dot(cerr_D,cerr_D);
    lgerr_D = dot(sigma,cerr_D);
    if nargout >= 3
        cjerr_D = sparser.compSprsVecTrnsp(cerr_D);
    end
end

function [lambda, lgerr, grad_x_const] = update_mlp(beta, cerr, cjerr,...
 lambda, grad_x_const, gradU)
    lambda.D = lambda.D + beta.D*cerr.D;
    lambda.A = lambda.A + beta.A*cerr.A;
    lgerr = struct('D', dot(lambda.D,cerr.D),'A', dot(lambda.A,cerr.A));
    grad_x_const.A = grad_x_const.A + beta.A*cjerr.A;
    grad_x_const.D = grad_x_const.D + beta.D*cjerr.D;
    grad_x_const.sum = gradU;
end
    
function [beta, changed] = update_beta(beta, opts, grad_x_const, cerr)
    prev_beta = beta;
    if nargin < 3
        beta.D = beta.D*opts.rate_ctn;
        beta.A = beta.A*opts.rate_ctn;
        beta.scldA = beta.scldA*opts.beta_rate_ctn;
    else
        if beta.A*norm(cerr.A,inf) < opts.beta_rate_thresh * norm(grad_x_const.A,inf)
            beta.A = beta.A*opts.beta_rate_ctn;
            beta.scldA = beta.scldA*opts.beta_rate_ctn;
        end
        if beta.D*norm(cerr.D,inf) < opts.beta_rate_thresh * norm(grad_x_const.D,inf)
            beta.D = beta.D*opts.beta_rate_ctn;
        end
    end
    beta = restrict_beta(beta);
    changed = struct('D',(beta.D ~= prev_beta.D),... 'Z',(beta.Z ~= prev_beta.Z),...
        'A',(beta.A ~= prev_beta.A));
end


