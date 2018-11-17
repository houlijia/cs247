function opts = removeCommonFields(opts)
  % removeCommonFieldsr removes fields which are identical
  % 
  % opts is a struct array. Fields in opts which are function handles or which
  % are identical in all structs are removed.
  
  if isempty(opts)
    opts = struct();
    return
  end
  
  flds = fieldnames(opts);
  for k=1:length(flds)
    fld = flds{k};
    if isa(opts(1).(fld), 'function_handle')
      opts = rmfield(opts, fld);
    else
      all_eql = true;
      for j=2:numel(opts)
        if ~isEqual(opts(1).(fld), opts(j).(fld))
          all_eql = false;
          break;
        end
      end
      if all_eql
        opts = rmfield(opts, fld);
      end
    end
  end
  
  flds = fieldnames(opts);
  for k=1:length(flds)
    fld = flds{k};
    if ~isstruct(opts(1).(fld))
      continue;
    end
    vals = [opts.(fld)];
    if isequal(size(vals), size(opts))
      vals = removeCommonFields(vals);
      for j=1:numel(opts)
        opts(j).(fld) = vals(j);
      end
    elseif isempty(vals)
      opts = rmfield(opts, fld);
    end
  end
end

