function mex_clnp = initMexContext()
  % initMexContext creates MexContext if necessary. If it was necessary to
  % create MexContext, the function returns a cleanup object that, when
  % deleted, will destroy the MexContext by calling deleteMexContext().
  % Otherwise the function returns an empty array.
  global mex_context mex_context_val;
  
  if isempty(mex_context);
    mex_clnp = onCleanup(@deleteMexContext);
    mex_context = MexContext();
    mex_context_val = mex_context.mx_cntxt;
  else
    mex_clnp = [];
  end


end

