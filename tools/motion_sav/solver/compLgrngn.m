% Compute the value of the Lagrangian. Note that if
% lmderr is specified, lambda is not necessary.
function [lgr,sqrerr, lmderr] = compLgrngn(beta, lambda, xerr, ...
        lmderr, sqrerr)
    if nargin < 5 || isempty(sqrerr)
        sqrerr = struct('A', dot(xerr.A,xerr.A), 'D', dot(xerr.D,xerr.D));
    end
    if nargin < 4 || isempty(lmderr)
        lmderr = struct('A', dot(lambda.A, xerr.A),...
            'D', dot(lambda.D, xerr.D));
    end
    lgr = xerr.J + lmderr.D + (beta.D*0.5)*sqrerr.D +...
        lmderr.A + (beta.scldA*0.5)*sqrerr.A ;
end


