function  prof_ctrl(code_type, prof_spec)
  % prof_ctrl suspends or resumtes profiling
  %
  % Input arguments:
  %   code_type: the type of code that follows (0-2):
  %     0 - code should never be profiled
  %     1 - code should be prfiled when profiling everything
  %     2 - code is important - should be profiled whenever profiling is done.
  %   prof_spec: (optional, default: 0) the specified profiling level (0-2):
  %     0 - no profiling is done
  %     1 - profile only important stuff
  %     2 - profile everything
  %    If prof_spec is a struct, the function looks for the fild 'prof_spec' in
  %    prof_spec
  
  if nargin >= 2 && isstruct(prof_spec) && isfield(prof_spec, 'prof_spec')
    prof_spec = prof_spec.prof_spec;
  else
      prof_spec = 0;
  end
  if code_type + prof_spec > 2
    profile resume;
  else
    profile off;
  end
end

