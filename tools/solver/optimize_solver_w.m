function [xerr_W, J_opt, wvec, Dxvec] = optimize_solver_w(xvec,...
        sparser, beta_W, lambda_W, wvec_ref)

    Dxvec = sparser.compSprsVec(xvec);

    % Compute the sparse vector which optimizes the lagrangian, given x, beta_D
    % and labmda_D.
    if nargin < 5 || isempty(wvec_ref)
        wvec = sparser.optimize(Dxvec, lambda_W, beta_W);
    else
        wvec = wvec_ref;
    end
    J_opt = sum(abs(wvec));
    
    xerr_W = Dxvec-wvec;
end
