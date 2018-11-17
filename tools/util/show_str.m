% show_str - generate a string representation of Matlab objects created
%   from the primitive numeric, logical and character types and any
%   combination of arrays, structs and cells.
%   Input arguments:
%     x - object to be represented
%     fmt - (optional) a struct giving the format to use. possible entries
%           are:
%             s - format for strings (default %s)
%             r - format for real numbers (default %.4g)
%             i - format for integers (default %d)
%             l - format for logical (as a number)
%     params - (optional) a struct of parameters controlling the operation:
%             prefix - prefix to add after each new line (default: '')
%             cell_marked - if true omit cell< > for top level cell
%             struct_marked - if true omit  struct for top level struct
%             struct_sep - separator between struct fields. Default: \n
%             (new line).

function s = show_str(x, fmt, params)
  if nargin < 3
    params = struct();
    if nargin <2
      fmt = struct('s','%s','r','%.4g','i','%d');        
    end
  end
  if ~isfield(fmt,'s')
    fmt.s = '%s';
  end
  if ~isfield(fmt,'r')
    fmt.r = '%.4g';
  end
  if ~isfield(fmt,'i')
    fmt.i = '%d';
  end
  
  if ~isfield(params,'cell_marked')
    params.cell_marked = false;
  end
  if ~isfield(params, 'prefix')
    params.prefix = '';
  end
  if ~isfield(params,'struct_marked')
    params.struct_marked = false;
  end
  if ~isfield(params, 'struct_sep')
    params.struct_sep = '\n';
  end
    
  if isstruct(x)
    names = fieldnames(x);
    if ~params.struct_marked
      prms = params;
      prms.struct_marked = true;
      if isempty(names)
        s ='struct<>';
      elseif isscalar(x)
        if strcmp(prms.struct_sep, '\n')
          s = sprintf('struct<\n%s >', show_str(x,fmt,prms));
        else
          s = sprintf('struct< %s >', show_str(x,fmt,prms));
        end
      else
        if strcmp(prms.struct_sep, '\n')
          s = sprintf('struct(%s)<\n%s >', ...
            show_str(size(x)), show_str(x,fmt,prms));
        else
          s = sprintf('struct(%s)< %s >', ...
            show_str(size(x)), show_str(x,fmt,prms));
        end
      end
    elseif isempty(names)
        s = '';
    else
      params.struct_marked = false;
      flds = struct2cell(x);
      elmnts = cell(2*length(names),1);
      prms = params;
      prms.prefix = [prms.prefix '  '];
      prms.cell_marked = false;
      elmnts{1} = '';
      if strcmp(prms.struct_sep, '\n')
        elmnts(1:2:length(elmnts)) = {sprintf([prms.struct_sep '%s'], prms.prefix)};
        elmnts{1} = elmnts{1}(2:end);  % Remove first newline
      else
        elmnts(3:2:length(elmnts)) = {sprintf('%s', prms.struct_sep)};
        elmnts{1} = '';
      end
      if isscalar(x)
        for k=1:length(names)
          elmnts{2*k} = [names{k} ':' show_str(flds{k,1},fmt, prms)];
        end
      else
        sz = size(flds);
        sz1 = sz(2:end);
        instep = sz(1);
        if length(sz1) == 1
          sz1 = [1 sz];
        end
        for k=1:length(names)
          elmnts{2*k} = [names{k} ':' ...
            show_str(reshape(flds(k:instep:end),sz1),fmt,prms)];
        end
      end
      s = horzcat(elmnts{:});
    end
  elseif ~params.cell_marked && iscell(x)
    params.cell_marked = true;
    s = ['cell< ' show_str(x,fmt,params) ' >'];
  elseif ischar(x) && isrow(x)
    s = ['''' sprintf(fmt.s,x) ''''];
  elseif isobject(x) && ~ismethod(x, 'show_str')
    mc = metaclass(x);
    mcp = mc.PropertyList;
    prop_names = cell(size(mcp));
    n_names = 0;
    for k=1:numel(mcp)
      if mcp(k).Constant || mcp(k).Abstract || ...
          ~strcmp(mcp(k).GetAccess, 'public') || mcp(k).Hidden
        continue;
      end
      n_names = n_names+1;
      prop_names{n_names} = mcp(k).Name;
    end
    if ~params.struct_marked
      prms = params;
      prms.struct_marked = true;
      if isscalar(x)
        s = sprintf('%s<%s>', class(x), show_str(x,fmt,prms));
      else
        s = sprintf('%s(%s)<%s>', class(x),  show_str(size(x),fmt,prms), ...
          show_str(x,fmt,prms));
      end
    else
      if n_names == 0
        s = '';
      else
        params.struct_marked = false;
        params.prefix = [params.prefix '  '];
        strs = cell(2*n_names+1,1);
        if strcmp(params.struct_sep, '\n')
          strs(1:2:length(strs)-2) = {sprintf(params.struct_sep)};
          strs(length(strs)) = {sprintf([params.struct_sep params.prefix])};
        else
          strs{1} = '';
          strs(3:2:length(strs)) = {sprintf(params.struct_sep)};
        end
        if isscalar(x)
          for k=1:n_names;
            strs{2*k} = [params.prefix prop_names{k} ': ' ...
              show_str(x.(prop_names{k}), fmt, params)];
          end
          s = horzcat(strs{:});
        else
          elmnts = cell(size(x));
          for j=1:numel(x)
            for k=1:n_names;
              strs{2*k} = [params.prefix prop_names{k} ': ' ...
                show_str(x.(prop_names{k}), fmt, params)];
            end
            elmnts{j} = [ '{ ' horzcat(strs{:}) ' }' ];
          end
          params.cell_marked = true;
          s = show_str(elmnts, fmt, params);
        end
      end
    end
  elseif isempty(x)
    s = '[]';
  elseif isscalar(x)
    if iscell(x)
      s = ['{' show_str(x{1},fmt,params) '}'];
    elseif isnumeric(x)
      if ~isreal(x)
        s=[show_str(real(x),fmt) '+' show_str(imag(x),fmt) 'i'];
      elseif ~isfinite(x)
        if isnan(x)
          s = 'NaN';
        else
          switch x
            case inf
              s='inf';
            case -inf
              s='-inf';
            otherwise
              error('x is not finite, not inf and not NaN');
          end
        end
      else
        if mod(x,1)
          s = sprintf(fmt.r, x);
        else
          s = sprintf(fmt.i, x);
        end
      end
    elseif islogical(x)
      if ~isfield(fmt,'l')
        if x
          s = 'T';
        else
          s = 'F';
        end
      else
        s = sprintf(fmt.l, x);
      end
    elseif isa(x, 'function_handle')
      s = func2str(x);
    else
      s = '??';
    end
  else
    sz = size(x);
    if isvector(x)
      brckt = {'',''};
      if isrow(x)
        separator = ',';
      else
        separator = ';';
      end
      outstep = 1;
      instep = length(x);
      step = 1;
      nl = length(x);
      sz1 = [1 1];
    elseif length(sz) == 2
      brckt = {'',''};
      separator = ';';
      outstep = 1;
      instep = sz(1);
      step = numel(x);
      nl = sz(1);
      sz1 = [1 sz(2)];
    else
      brckt = {'[',']'};
      separator = ',';
      instep = 1;
      sz1 = sz(1:end-1);
      outstep = prod(sz(1:end-1));
      step = outstep;
      nl = sz(end);
    end
    elmnts = cell(1,2*nl-1);
    elmnts(2:2:end) = {separator};
    ofst = 0;
    for k=1:nl
      elmnts{2*k-1} = [brckt{1} ...
        show_str(reshape(x(ofst+1:instep:ofst+step), sz1), ...
        fmt, params) brckt{2}];
      ofst = ofst + outstep;
    end
    s = sprintf('[%s]', horzcat(elmnts{:}));
  end
end
