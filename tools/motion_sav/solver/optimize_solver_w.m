function [xerr_D, J_opt, wvec, Dxvec] = optimize_solver_w(xvec,...
        sparser, beta_D, lambda_D, wvec_ref)

    Dxvec = sparser.compSprsVec(xvec);

    % Compute the sparse vector which optimizes the lagrangian, given x, beta_D
    % and labmda_D.
    if nargin < 5 || isempty(wvec_ref)
        wvec = sparser.optimize(Dxvec, lambda_D, beta_D);
    else
        wvec = wvec_ref;
    end
    J_opt = sum(abs(wvec));
    
    xerr_D = Dxvec-wvec;
end
