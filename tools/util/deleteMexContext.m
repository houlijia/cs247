function deleteMexContext()
  % deleteMexContext clears the mex context. 

  global mex_context mex_context_val;
  if ~isempty(mex_context)
    mex_context.delete();
    mex_context = [];
    mex_context_val = [];
  end

end

