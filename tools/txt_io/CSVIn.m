classdef CSVIn < TextIn & CSVIo
    % CSVIn - input of records as CSV files
    
    properties
    end
    
    methods
        % Constructor. Same arguments as for superclass CSVIo
        function obj = CSVIn()
            obj = obj@TextIn();
            obj = obj@CSVIo(cell(2,0), ''); % Dummy generation
            
            % read and process header
            lines = obj.readLines(3);
            [flds, nm, nrec] = CSVIo.parseHeader(lines);
            obj.set(flds, nm, nrec);
         end
        
        % Read lines and clean from comments, trailing and leading blanks
        % and empty lines, until n_lines have accumulated.
        % Input:
        %   obj - this object
        %   n_lines - (optionl) number of clean lines to read. Default=1
        % Output
        %   lines - a cell array of n_lines lines (strings)
        %   cnt - total number of bytes read
        %   emsg - if not empty, an error string. If missing error will
        %          cause an exception.
        function [lines, cnt, emsg] = readLines(obj, n_lines)
            if nargin < 2
                n_lines = 1;
            end
            
            lines = cell(n_lines,1);
            indx = 0;
            cnt = 0;
            try
                while indx < n_lines
                    line = cell(n_lines-indx,1);
                    for k=1:length(line)
                        [line{k}, emsg, n] = obj.read();
                        if ~isempty(emsg)
                            error('reading failed (%s)',emsg);
                        end
                        cnt = cnt + n;
                    end
                    line = CSVIo.removeComments(line);
                    for k=1:length(line)
                        indx = indx+1;
                        lines{indx} = line{k};
                    end
                end
            catch err
                if nargout < 3
                    rethrow(err);
                else
                    emsg = err.message;
                end
            end
            
            obj.byte_cnt = obj.byte_cnt + cnt;
        end
       
        % Read nrec records as a cell array of (nrec,no. fields).
        % If not specified nrec = 1.
        function [records, cnt, emsg] = readRecordAsCell(obj, nrec)
            if nargin < 2
                nrec = 1;
            end
            
            try
                [lines, cnt, emsg] = obj.readLines(nrec);
                if ~isempty(emsg)
                    error('%s',emsg);
                end
                
                records = obj.str2cell(lines);
                obj.rec_left = obj.rec_left - nrec;
            catch err
                if nargout < 3
                    rethrow(err);
                else
                    emsg = err.message;
                end
            end
        end
            
        % Read nrec records as a struct array of nrec entries.
        % If not specified nrec = 1.
        function [records, cnt, emsg] = readRecordAsStruct(obj, nrec)
            if nargin < 2
                nrec = 1;
            end
            try
                [records, cnt, emsg] = obj.readRecordAsCell(nrec);
                if ~isempty(emsg)
                    error('%s',emsg);
                end
                records = cell2struct(records, obj.fields(1,:),2);
            catch err
                if nargout < 3
                    rethrow(err)
                else
                    emsg = err.message;
                end
            end
        end
        
    end
end

