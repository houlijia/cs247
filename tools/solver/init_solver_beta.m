% Initialize beta
function beta = init_solver_beta(opts, sens_mtrx, sparser)
%     % Third argument is a place holder for sparser
%      if ~isempty(sparser.expndr)
%       bscl = sparser.expndr.norm();
%     else
%       bscl = 1;
%     end
  a_scl = sens_mtrx.norm()^2;
  d_scl = sparser.norm()^2;
  beta = struct('A',opts.beta_A0/a_scl, 'D',opts.beta_D0/d_scl,...
  'final',struct('A',opts.beta_A/a_scl, 'D',opts.beta_D/d_scl));
end



