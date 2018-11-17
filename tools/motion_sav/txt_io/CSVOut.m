classdef CSVOut < TextOut & CSVIo
    % CSVOut - output of records as CSV files
    
    properties
    end
    
    methods
        % Constructor. Same arguments as for superclass CSVIo
        function obj = CSVOut(flds, nm, nrec)
            if nargin <3
                nrec = inf;
            end
            obj = obj@TextOut();
            obj = obj@CSVIo(flds, nm, nrec);
            
            hdr = obj.getHeader();
            cnt = obj.write(hdr);
            if ischar(cnt)
                error('Failed to write header of CSVOut %s: %s',...
                    nm, cnt);
            end
            obj.byte_cnt = cnt;
        end
        
        % Change the parameters and write a new header into the same file.
        function cnt = setCSVOut(obj, flds, nm, nrec)
            if nargin < 4
                nrec = inf;
            end
            
            try
                if obj.rec_left < inf && obj.rec_left > 0
                    error('Need to write %d more records', obj.rec_left)
                end
                
                obj.set(flds, nm, nrec);
                hdr = obj.getHeader();
                cnt = obj.write(hdr);
                if ischar(cnt)
                    error('Failed to write header of CSVOut %s: %s',...
                        nm, cnt);
                end
                obj.byte_cnt = obj.byte_cnt + cnt;
            catch err
                if nargout == 0
                    rethrow(err);
                else
                    cnt = err.message;
                end
            end
        end
        
        % Record can be a cell array or struct
        function cnt = writeRecord(obj,record)
            try
                lines = obj.record2line(record);
                if length(lines) > obj.rec_left
                    error('Writing into %s more records than %d allowed',...
                        obj.name, obj.n_records);
                end
                for i=1:length(lines)
                    cnt = obj.writeLine(lines{i});
                    if ~ischar(cnt)
                        obj.byte_cnt = obj.byte_cnt + cnt;
                        obj.rec_left = obj.rec_left - 1;
                    elseif nargout == 0
                        error('Failed to write record of type %s: %s',...
                            obj.name, cnt);
                    end
                end
            catch err
                if nargout == 0
                    rethrow(err);
                else
                    cnt = err.message;
                end
            end
        end
    end
end

