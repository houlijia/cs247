classdef InputFilesDef
    %InputFilesDef Summary Each objects of this class defines a list of
    %input files
    %   Detailed explanation goes here
    
    properties (SetAccess=immutable)
        files;
    end
    
    methods
        function files_def = InputFilesDef(spec)
            % InputFilesDef.InputFilesDef - Summary Constructor, builds
            % files[] from spec.
            %
            % Input:
            % spec can be one of the following
            %     An cell(1,n) containing a list of files (full path)
            %     A JSON string which specifies an array called "names" and
            %            optionally a string called "base_dir"
            %     A string beginning with "<", inerpreted as the name of a
            %     file containing the JSON string.
            
            if iscell(spec)
                % spec is a cell array of file names. Use it as is
                sz = size(spec);
                szsz = size(sz);
                if (szsz(2) ~= 2) || (sz(1) ~= 1)
                    exc = MException('InputFilesDef:InputFileDef:BadCellSize',...
                        'Illegal size of input cell: % d', sz);
                    throw(exc)
                end
                files_def.files = spec;
            elseif ischar(spec)
                % spec is a string.  If it starts with '<' read a JSON
                % string from the specified file, else use spec as 
                % the JSON string
                if strcmp(spec(1),'<')
                    jstr = fileread(spec(2:end));
                else
                    jstr = spec;
                end
                info = parse_json(jstr);
                
                % If a base dir was specified use it; otherwise ignore it
                if isfield(info, 'base_dir')
                    if ~ strcmp(info.base_dir(end), '/')
                        info.base_dir = [info.base_dir '/'];
                    end
                else
                    info.base_dir = '';
                end
                
                files_def.files = strcat(info.base_dir, info.names); 
            else
                exc = MException('InputFilesDef:InputFileDef:BadInputType',...
                        'Illegal argument to constructor');
                    throw(exc)
            end
            
        end  % InputFilesDef
    end   % Methods
    
end

