function eql = isEqual(x,y)
  eql = false;
  
  if ~strcmp(class(x), class(y))
    return;
  end
  if isempty(x) && isempty(y)
    eql = true;
    return
  elseif  ~isequal(size(x), size(y))
    return
  end
  
  if isa(x,'function_handle')
    eql = true;
    return
  elseif isstruct(x)
    fldx = fieldnames(x);
    fldy = fieldnames(y);
    if ~isequal(fldx, fldy)
      return
    end
    
    for f=1:numel(fldx)
      fld = fldx{f};
      for k=1:numel(x)
        if ~isEqual(x(k).(fld),y(k).(fld))
          return
        end
      end
    end
  elseif iscell(x)
    for k=1:numel(x)
      if ~isEqual(x{k},y{k})
        return
      end
    end
  elseif ~isa(x,'gpuArray') && isobject(x) && ...
      (isa(x,'CodeElement') || ismethod(x, 'isEqual'))
    for k=1:numel(x)
      if ~x(k).isEqual(y(k))
        return
      end
    end
  elseif ~isequal(x,y);
    return
  end
  
  eql = true;
end

