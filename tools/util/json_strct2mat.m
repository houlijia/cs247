function arg = json_strct2mat(arg)
  % Reverse the action of json_mat2strct()
  function ok = is_mat_strct(arg)
    if isstruct(arg) && isfield(arg, 'A_')
      switch length(fieldnames(arg))
        case 1
          ok = true;
        case 2
          ok = isfield(arg,'C_') || isfield(arg,'S_');
        case 3
          ok = isfield(arg,'C_') && isfield(arg,'S_');
        otherwise
          ok = false;
      end
    else
      ok = false;
    end
  end
  
  if iscell(arg)
    for k=1:numel(arg)
      arg{k} = json_strct2mat(arg{k});
    end
  elseif is_mat_strct(arg)
    if isempty(arg.A_)
      val = [];
    elseif isstruct(arg.A_{1})
      flds = fieldnames(arg.A_{1});
      val = cell([length(flds) size(arg.A_)]);
      val = cell2struct(val, flds, 1);
      for k=1:numel(arg.A_)
        val(k) = json_strct2mat(arg.A_{k});
      end
    else
      val = cell2mat(arg.A_);
    end
    if isfield(arg, 'S_')
      siz = cell2mat(arg.S_);
      val = reshape(val, siz);
    else
      val = val(:)';  % By default a row vector
    end
    if isfield(arg, 'C_')
      val = cast(val, arg.C_);
    end
    arg = val;
  elseif isstruct(arg)
    flds = fieldnames(arg);
    for k=1:length(flds)
      fld = flds{k};
      arg.(fld) = json_strct2mat(arg.(fld));
    end
  end
end
