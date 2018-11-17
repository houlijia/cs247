% Initialize beta
function beta = init_solver_beta(opts, sens_mtrx, ~)
    % Third argument is a place holder for sparser
    beta = struct('A',opts.beta_A0/sens_mtrx.normAtA(),...
        'D',opts.beta_D0,...
        'final',struct('A',opts.beta_A/sens_mtrx.normAtA(),...
        'D',opts.beta_D));
    beta.scldA = beta.A * sens_mtrx.trnspScale();
    beta.final.scldA = beta.final.A * sens_mtrx.trnspScale();
end



