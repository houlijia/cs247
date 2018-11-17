% Converts a struct into an arguments list
% Input:
%   args - the original argument list (cell array)
%   names - a cell array of arg names (char strings)
% Output
%   args - the resulting argument list
% 
% args is modified only if it has one element (one argument) and that
% element is either a character string or a struct. If the element is a
% character string it is first converted to a struct usring
% ProcessingParams.parse_opts (as a JSON string or a file containing a JSON
% string).  Once the struct is available, args is created by matching
% elements in names with the fields of the struct, until we reach a name
% for which there is no corresponding field.

function args = parseInitArgs(args, names)
  if length(args)==1 && (isstruct(args{1}) || ischar(args{1}))
    if ischar(args{1})
      spec = ProcessingParams.parse_opts(args{1});
    else
      spec = args{1};
    end
    n_names = length(names);
    args = cell(1,n_names);
    for k=1:n_names
      if isfield(spec, names{k})
        args{k} = spec.(names{k});
      else
        args = args(1:k-1);
        break;
      end
    end
  end
end

