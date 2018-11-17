function s = show_str(x, fmt, params)
  if nargin < 3
    params = struct();
    if nargin <2
      fmt = '';
    end
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
    
  if isstruct(x)
    names = fieldnames(x);
    if ~params.struct_marked
      prms = params;
      prms.struct_marked = true;
      if isempty(names)
        s ='struct<>';
      elseif isscalar(x)
        s = sprintf('struct<\n%s >', show_str(x,fmt,prms));
      else
        s = sprintf('struct(\n%s)<%s >', ...
          show_str(size(x)), show_str(x,fmt,prms));
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
      elmnts(1:2:length(elmnts)) = {sprintf('\n%s', prms.prefix)};
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
            show_str(reshape(flds(k:instep:end),sz1),fmt),prms];
        end
      end
      s = horzcat(elmnts{:});
      s = s(2:end);  % Remove first newline
    end
  elseif ~params.cell_marked && iscell(x)
    params.cell_marked = true;
    s = ['cell< ' show_str(x,fmt,params) ' >'];
  elseif ischar(x) && isrow(x) && (isempty(fmt) || strcmp(fmt(end),'s'))
    if isempty(fmt)
      fmt = '%s';
    end
    s = ['''' sprintf(fmt,x) ''''];
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
        if isempty(fmt)
          if mod(x,1)
            fmt = '%.4f';
          else
            fmt = '%d';
          end
        end
        s = sprintf(fmt, x);
      end
    elseif islogical(x)
      if isempty(fmt)
        if x
          s = 'T';
        else
          s = 'F';
        end
      else
        s = sprintf(fmt, x);
      end
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
