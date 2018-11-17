classdef CSVIo < handle
    %CSVIo is a base class for input and output of records of comma
    %separated values (CSV)
    %   Detailed explanation goes here
    
    properties (Constant)
        % Format for first line
        start_line_fmt = '****==== name ====****';
    end
    
    properties
        % a cell array of 2xN. field{1,k} is the name of the k-th field
        % and field{2,k} is the type of the data in the field. The types
        % are:  'T' - text, 'I' - integer, 'F' - float.
        fields; 
        
        % If not empty, type of the record
        name = '';
        
        n_records = inf;  % number of records
        rec_left = inf;
        
        byte_cnt = 0;  % number of bytes read or written
    end
    
    methods
        % Constructor
        %  flds - value for fields
        %  nm - name of the record
        %  nrec - (optional) number of records
        function obj = CSVIo(flds, nm, nrec)
            if nargin > 1
                if nargin <= 2
                    nrec = inf;
                end
                obj.set(flds, nm, nrec);
            end
        end

        % return a string containing the 3 lines of header (no newline at
        % the end of the last line)
        function header=getHeader(obj)
            if obj.n_records == inf
                nm = obj.name;
            else
                nm = sprintf('%s (%d)',obj.name, obj.n_records);
            end
            start_line = regexprep(obj.start_line_fmt, 'name', nm);
            flds = obj.fields;
            for k=1:size(flds,2)
                if isempty(regexp(flds{2,k},'[TIF]','once'))
                    error('Illegal field type (%d): %s',k, flds{2,k});
                end
            end
            for k=1:(size(flds,2)-1)
                flds{1,k} = [flds{1,k} ','];
                flds{2,k} = [flds{2,k} ','];
            end
            header = sprintf('%s\n%s\n%s', start_line,...
                horzcat(flds{1,:}), horzcat(flds{2,:}));
        end
         
        % Set based on header specified by a cell array of 3 line strings 
        % or a single string with newlines
        % string
        function emsg = setHeader(obj,lines)
            [flds, nm, nrec, emsg] = obj.parseHeader(lines);
            if ~isempty(emsg)
                if ~nargout
                    error('%s',emsg);
                else
                    return
                end
            end
            obj.set(flds, nm, nrec);
        end
            
        function set(obj, flds, nm, nrec)
            % Set values
            obj.fields = flds;
            obj.name = nm;
            obj.n_records = nrec;
            obj.rec_left = nrec;
        end
        
        % Convert a record to a text line. 
        % Record can be a struct or a cell array
        % Output (lines) is a cell array of strings cotnaining the lines
        function lines = record2line(obj, record)
            if isstruct(record)
                rec = cell(length(record),length(obj.fields));
                for i=1:length(record)
                    for k=1:length(obj.fields)
                        rec{i,k} = record(i).(obj.fields{1,k});
                    end
                end
                record = rec;
            end
            lines = cell(size(record,1),1);
            for i=1:size(record,1)
                for k=1:size(record,2)
                    switch obj.fields{2,k}
                        case 'I'
                            record{i,k} = sprintf('%d,',record{i,k});
                        case 'F'
                            if record{i,k} == floor(record{i,k})
                                record{i,k} = sprintf('%d,',record{i,k});
                            else
                                record{i,k} = sprintf('%f,',record{i,k});
                            end
                        case 'T'
                            % prepend '\' to ',' and '\' in fields and append ,
                            record{i,k} = ...
                                [regexprep(record{i,k},'([,\\])','\\$1') ','];
                    end
                end
                line = horzcat(record{i,:});

                % remove trailing comma and prepend (append) '|' to 
                % whitespace or '|' at the beginning(end)
                line = regexprep(line(1:end-1),'^([|\s])','|$1');
                lines{i} = regexprep(line,'([|\s])$','$1|');
            end
        end
        
        % Convert a string (one or more lines) into a cell array.  The first
        % dimension of the cell array is the number of lines; the second is
        % the number of fields
        function records = str2cell(obj, str)
            lines = obj.removeComments(str);
            records = cell(length(lines),size(obj.fields,2));
            
            for i = 1:length(lines)
                % Remove '|' at the beginning and end
                lines{i} = regexprep(lines{i}, '((^[|])|([|]$))','');
                
                % Break into fields using ',' as separator, but ignoring
                % commas preceded by an even number of '\'
                flds = regexp(lines{i},...
                    '(((\\\\)*([^,\\]|(\\[^\\]))?)*)(?:,|$)','tokens');
                for k = 1:length(flds)
                    switch obj.fields{2,k}
                        case 'I'
                            [records{i,k},cnt] = sscanf(flds{k}{1},'%d');
                        case 'F'
                            [records{i,k},cnt] = sscanf(flds{k}{1},'%f');
                        case 'T'
                            % replace '\\' and '\,' by '\' and ',',
                            % respectively
                            records{i,k} = regexprep(flds{k}{1}, '\\([\\,])','$1');
                            cnt = 1;
                    end
                    if ~cnt
                        error('failed to parse field %s into (%d,%d) as %s',...
                        fld, i, k, obj.fields{2,k});
                    end
                end
            end
        end
        
        function records = str2struct(obj, str)
            records = cell2struct(obj.str2cell(str), obj.fields, 2);
        end
        
    end
    
    methods (Static)
        % Remove leading and trailing blanks and comments (beginning with #),
        % and blank lines.
        % Input may be a either a string or a cell array with individual
        % lines. 
        % Output is a cell array of individual lines
        function clean_str = removeComments(str)
            if ischar(str)
                str = regexp(str, '\n', 'split');
            end
            clean_str = cell(size(str));
            n_lines = 0;
            
            for k=1:length(str)
                % Remove comments and trailing blanks
                line = regexprep(str{k}, '\s*(#.*)?$','');
                
                % remove leading blanks
                line = regexprep(line, '^\s*', '');
                
                if ~isempty(line)
                    n_lines = n_lines+1;
                    clean_str{n_lines} = line;
                end
            end
            clean_str = clean_str(1:n_lines);
        end
        
        % Tries to parse input string (line) according to start_line_fmt.
        % If successful, rerturns name as a string (may be empty).
        % Otherwise returns numeric 0. nrec is the number of records, 
        % In case of error name is numeric 0
        function [name, nrec] = parseStartLine(line)
            expr = regexprep(CSVIo.start_line_fmt, '[*]', '\\*');
            expr = regexprep(expr, '\sname\s','\s*(.*)\s*');
            namecell = regexp(line, expr, 'tokens');
            if isempty(namecell)
                name = 0;
                return
            end
            name = namecell{1}{1};
            tokens = regexp(name,  '\s*\((\d+)\)\s*$','tokens');
            if ~isempty(tokens)
                nrec = sscanf(tokens{1}{1},'%d');
                name =regexprep(name, '\s*\(\d+\)\s*$','');
            end
        end
                
        % Parse 3 lines containing the header and determine the parameters
        % of the CSVIo objects. str can be a cell of 3 strings, each
        % representing a line, or one long string conaining all three
        % lines.  If there is an error, it is returned in emsg (otherwise
        % empty).
        function [fields, name, nrec, emsg] = parseHeader(str)
            emsg = '';
            try
                lines = CSVIo.removeComments(str);
                if length(lines) > 3
                    error('Too many (%d) lines in header:\n%s',...
                        length(lines), str);
                end
                
                [name, nrec] = CSVIo.parseStartLine(lines{1});
                if ~ischar(name)
                    error('First line not in correct format: "%s"',...
                        lines{1});
                end
                
                flds1 = regexp(lines{2},',','split');
                flds2 = regexp(lines{3},',','split');
                
                fields = cell(2,length(flds1));
                fields(1,:) = flds1(:);
                fields(2,:) = flds2(:);
                
            catch err
                if nargout < 4
                   rethrow(err);
                else
                    emsg = err.message;
                end
            end
            
        end

    end
    
end

