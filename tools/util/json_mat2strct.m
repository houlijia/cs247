function arg = json_mat2strct(arg)
  % JSON does not have a way to represent Matlab arrays, so it converts
  % them to cell arrays, which can create a problem in decoding. This
  % function converts each array into a struct with a field 'A_', so on
  % decoding it can be recognized and decoded correctly.
  if (~ischar(arg) && ~isscalar(arg)) || iscell(arg)
    if iscell(arg)
      for k=1:numel(arg)
        arg{k} = json_mat2strct(arg{k});
      end
    else
      if isstruct(arg)
        for k=1:numel(arg)
          arg(k) = json_mat2strct(arg(k));
        end
      end
      val = struct('A_', arg(:));
      if length(size(arg)) > 2 || size(arg,1) ~= 1
        val.S_ = size(arg);
      end
      if ~isa(arg,'double') && ~isstruct(arg)
        val.C_ = class(arg);
      end
      arg = val;
    end
  elseif isstruct(arg)
    flds = fieldnames(arg);
    for j=1:length(flds);
      fld = flds{j};
      arg.(fld) = json_mat2strct(arg.(fld));
    end
  end
end
