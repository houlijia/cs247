function J=mat2json(M,F,cmpct)

%JSON2MAT converts a Matlab structure into a javscript data object (JSON).
%         M can also be a file name. In the spirit of fast prototyping 
%         this function takes a very loose approach to data types and 
%         dimensionality - neither is explicitly retained.
%
%         The second input argument is optional and when used it indicates
%         the name of the file where J is to be stored (empty=ignore).
%
%         if cmpct is present and true, the JSON string will be compact - no 
%            newlines or spaces to help readability.
%
%Example: mat2json(json2mat('{lala:2,lele:4,lili:[1,2,{bubu:5}]}')) 
%
% Jonas Almeida, March 2010

    function J=do_mat2json(M, prefix)
        if ischar(M) || (numel(M) <= 1 && ~iscell(M))
            if isstruct(M)
                f=fieldnames(M);
                if ~isempty(f)
                    C = cell(size(f));
                    if ~cmpct
                      prfx = [prefix, '  '];
                      for i=1:length(f)
                        if i < length(f)
                          C{i} = sprintf('%s"%s":%s,\n', ...
                            prfx, f{i}, do_mat2json(M.(f{i}),prfx));
                        else
                          C{i} = sprintf('%s"%s":%s\n', ...
                            prfx, f{i}, do_mat2json(M.(f{i}),prfx));
                        end
                      end
                      J = sprintf('%s{\n%s%s}', prefix, [C{:}], prefix);
                    else
                      for i=1:length(f)
                        if i < length(f)
                          C{i} = sprintf('%s"%s":%s,', ...
                            prefix, f{i}, do_mat2json(M.(f{i}),prefix));
                        else
                          C{i} = sprintf('%s"%s":%s', ...
                            prefix, f{i}, do_mat2json(M.(f{i}),prefix));
                        end
                      end
                      J = sprintf('%s{%s%s}', prefix, [C{:}], prefix);
                    end
                else
                    J = '{}';
                end
            elseif isempty(M)
                J = 'null';
            elseif islogical(M)
                if M
                    J='true';
                else
                    J='false';
                end
            elseif isnumeric(M)
                J=num2str(M);
            elseif isobject(M) && ismethod(M,'getJSON')
              J = M.getJSON();
            elseif isa(M,'function_handle')
              J=['"@', func2str(M),'"'];
            else % Assume character string
                J = ['"',M,'"'];
            end
        elseif isempty(M)
            J='[]';    % empty cell
        else
            sz = size(M);
            C = cell(sz);
            if iscell(M)
                for i=1:numel(M)
                    C{i} = [do_mat2json(M{i}, prefix),','];
                end
            else
                for i=1:numel(M)
                    C{i} = [do_mat2json(M(i), prefix),','];
                end
            end
            
            if length(sz) == 2 && (sz(1)==1 || sz(2)==1)
                J = ['[', C{:}];
                J(end) = ']';
            else
                while length(sz) > 1
                    sz1 = sz(2:end);
                    C = reshape(C, [sz(1), prod(sz1)]);
                    D=cell(1,size(C,2));
                    for i=1:length(D)
                        D{i} = ['[', C{:,i}];
                        D{i} = [D{i}(1:end-1)  '],'];
                    end
                    C = D;
                    sz = sz1;
                end
                J = ['[', C{:}];
                J(end) = ']';
            end
        end
    end

    if nargin < 3
      cmpct = false;
    end
    J = do_mat2json(M,'');
    if nargin>1  && ~isempty(F) 
      %save JSON result in file
        if ischar(F)
            fid=fopen(F,'w');
        end
        fprintf(fid,'%s',J);
        if ischar(F)
            fclose(fid);
        end
    end
end
