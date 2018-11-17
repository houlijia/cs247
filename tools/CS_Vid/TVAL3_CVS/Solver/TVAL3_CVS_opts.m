function opts = TVAL3_CVS_opts(opts)
%
% Set default options for users' convenience.
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 06/05/2010


if isfield(opts,'mu')
    if ~isscalar(opts.mu) || opts.mu <0
        error('opts.mu must be positive.');
    end
else
    opts.mu = 2^7;
end
% The optimal mu is primarily depend on the noise level. 
% Use large mu when b is noise-free and small mu when b is noisy.


if isfield(opts,'beta')
    if ~isscalar(opts.beta) || opts.beta <0
        error('opts.beta must be positive.');
    end
else
    opts.beta = opts.mu/16;
end
% [u, b; mu, beta] is equivalent to [2u, 2b; mu/2, beta/2] ( the same solution after scaling). 


if isfield(opts,'tol')
    if ~isscalar(opts.tol) || opts.tol <= 0
        error('opts.tol should be a positive small number.');
    end
else
    opts.tol = 1.e-4;
end;
% outer loop tolerence


if isfield(opts,'tol_inn')
    if ~isscalar(opts.tol_inn) || opts.tol_inn <= 0
        error('opts.tol_inn should be a positive small number.');
    end
else
    opts.tol_inn = opts.tol;
end;
% inner loop tolerence


if isfield(opts,'maxcnt')
    if ~isscalar(opts.maxcnt) || opts.maxcnt <= 0
        error('opts.maxcnt should be a positive integer.');
    end
else
    opts.maxcnt = 30;
end
% max # of outer iterations


if isfield(opts,'maxin')
    if ~isscalar(opts.maxin) || opts.maxin <= 0
        error('opts.maxin should be a positive integer.');
    end
else
    opts.maxin = 15;
end
% max # of inner iterations


if isfield(opts,'maxit')
    if ~isscalar(opts.maxit) || opts.maxit <= 0
        error('opts.maxit should be a positive integer.');
    end
else
    opts.maxit = opts.maxin*opts.maxcnt;
end
% max # of total iterations


if isfield(opts,'init')
    if length(opts.init) ~= 1
        fprintf('User has supplied opts.init as initial guess matrix......\n');
    elseif ~isinInterval(opts.init,0,1,true) || opts.init ~= floor(opts.init)
        error('opts.init should be either 0/1 or an initial guess matrix.');
    end
else
    opts.init = 1;
end
% Specify the initial guess.


if isfield(opts,'StpCr')
    if ~isinInterval(opts.StpCr,0,1,true) || opts.StpCr ~= floor(opts.StpCr)
        error('opts.StpCr should be either 0 (optimality gap) or 1 (relative change).');
    end
else
    opts.StpCr = 0;
end
% Choose the stopping criteria between optimality gap or relative change.


if isfield(opts,'disp')
    if ~isscalar(opts.disp) || opts.disp <= 0
        error('opts.disp should be an integer.');
    end
else
    opts.disp = 0;
end


if isfield(opts,'scale_A')
    if ~islogical(opts.scale_A)
        error('opts.scale_A should be true or false.');
    end
else
    opts.scale_A = false;
end


if isfield(opts,'scale_b')
    if ~islogical(opts.scale_b)
        error('opts.scale_b should be true or false.');
    end
else
    opts.scale_b = false;
end


if isfield(opts,'consist_mu')
    if ~islogical(opts.consist_mu)
        error('opts.consist_mu should be true or false.');
    end
else
    opts.consist_mu = false;
end
% Decide if mu should be accordingly scaled while scaling A and b.
% Strongly recommend setting as 'false' if one try to recover a signal
% or a 3D video instead of solving an exact minimization problem.



if isfield(opts,'mu0')
    if ~isscalar(opts.mu0) || opts.mu0 <= 0
        error('opts.mu0 is should be a positive number which is no bigger than beta.');
    end
else
    opts.mu0 = 2^2;  
end
% initial mu


if isfield(opts,'beta0')
    if ~isscalar(opts.beta0) || opts.beta0 <= 0
        error('opts.beta0 is should be a positive number which is no bigger than beta.');
    end
else
    opts.beta0 = opts.mu0/16; 
end
% initial beta


if isfield(opts,'rate_ctn')
    if ~isscalar(opts.rate_ctn) || opts.rate_ctn <= 1
        error('opts.rate_ctn is either not a scalar or no bigger than one.');
    end
else
    opts.rate_ctn = 2;
end
% continuation parameter for both mu and beta

    
if isfield(opts,'c')
    if ~isscalar(opts.c) || opts.c <= 0 || opts.c > 1
        error('opts.c should be a scalar between 0 and 1.');
    end
else
    opts.c = 1.e-5;
end
% Ensure the sufficient decrease in nonmonotone Armijo condition.


if isfield(opts,'gam')
    if ~isscalar(opts.gam) || opts.gam <= 0 || opts.gam > 1
        error('opts.gam should be a scalar between 0 and 1.');
    end
else
    opts.gam = 0.99;
end
% Control the degree of nonmonotonicity. 0 corresponds to monotone line search.
% The best convergence is obtained by using values closer to 1 when the iterates
% are far from the optimum, and using values closer to 0 when near an optimum.


if isfield(opts,'rate_gam')
    if ~isscalar(opts.rate_gam) || opts.rate_gam <= 0 || opts.rate_gam > 1
        error('opts.rate_gam should be a scalar between 0 and 1.');
    end
else
    opts.rate_gam = .9;
end
% shrinkage rate of gam


if isfield(opts,'gamma')
    if ~isscalar(opts.gamma) || opts.gamma <= 0 || opts.gamma > 1
        error('opts.gamma should be a scalar between 0 and 1.');
    end
else
    opts.gamma = .6;
end
% backtracking rate


if isfield(opts,'nonneg')
    if ~islogical(opts.nonneg)
        error('opts.nonneg should be true or false.');
    end
else
    opts.nonneg = false;
end


if isfield(opts,'isreal')
    if ~islogical(opts.isreal)
        error('opts.isreal should be true or false.');
    end
else
    opts.isreal = false;
end


if isfield(opts,'TVL2')
    if ~islogical(opts.TVL2)
        error('opts.TVL2 should be true or false.');
    end
else
    opts.TVL2 = false;
end
% Decide the model: TV or TV/L2. 
% The default is TV model, which is recommended.


if isfield(opts,'aqst')
    if ~islogical(opts.aqst)
        error('opts.aqst should be true (data acquisition) or false (DCT-preprocessing).');
    end
else
    opts.aqst = false;
end
% Choose between data acquisition model and DCT-preprocessing model.
