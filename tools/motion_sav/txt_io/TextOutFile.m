classdef TextOutFile < TextOut
    %TextOutFile - TextOut for file output.
    
    properties
        fid = -1; % File identifier
        do_close=false; % logical. If true call fclose in destructor
    end
    
    methods
        % Constructor. fd can be a number (File ID) or a string (file
        % name).
        function obj = TextOutFile(fd)
            obj = obj@TextOut();
            if ischar(fd)
                [obj.fid, emsg] = fopen(fd, 'wt');
                if obj.fid == -1
                    error('Failed to open %s: %s', fd, emsg);
                end
                obj.do_close = true;
            else
                obj.fid = fd;
            end
        end
        
        % Destructor
        function delete(obj)
            if obj.do_close
                fclose(obj.fid);
            end
        end
        
        % Implementation of abstract function
        % Write out a line, appending newline. return number of chars
        % written, including the new line, or error string if an error
        % occurred
        %   cnt - number of chars written (including new line)
        function cnt = writeLine(obj,line)
            cnt = fprintf(obj.fid, '%s\n', line);
        end
    end
    
end

