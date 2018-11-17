function dup = duplicate( item, nocopy )
  %duplicate create a deep copy (as far as defined by copy methdos) of item
  
  if isobject(item)
    if ~isscalar(item)
      dup = item;
      for k=1:numel(item)
        dup(k) = duplicate(item(k));
      end
    elseif ismethod(item, 'copy') && (nargin<2 || ~nocopy)
      dup = item.copy();
    else
      dup = item;
      mc = metaclass(item);
      mcp = mc.PropertyList;
      for k=1:numel(mcp)
        if mcp(k).Constant || mcp(k).Abstract
          continue;
        end
        fld = mcp(k).Name;
        dup.(fld) = duplicate(item.(fld));
      end
    end
  elseif iscell(item)
    dup = cell(size(item));
    for k=1:numel(item)
      dup{k} = duplicate(item{k});
    end
  elseif isstruct(item)
    flds = fields(item);
    dup = duplicate(struct2cell(item));
    dup = cell2struct(dup,flds,1);
  else
    dup = item;
  end  
end

