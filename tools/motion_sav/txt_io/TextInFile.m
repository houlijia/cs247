classdef TextInFile < TextIn
    %TextInFile - TextIn for file input.
    
    properties
        fid = -1; % File identifier
        do_close=false; % logical. If true call fclose in destructor
    end
    
    methods
        % Constructor. fd can be a number (File ID) or a string (file
        % name).
        function obj = TextInFile(fd)
            obj = obj@TextIn();
            if ischar(fd)
                obj.fid = fopen(fd, 'rt');
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
        %   line - returned line (without terminating new line). -1 if
        %          EOF, 0 if error occurred.
        %   err - error string, empty if successful
        %   cnt - number of chars written (including new line), or -1
        %         if an error occurred 
        %   
        function [line, err, cnt] = read(obj)
            line = fgets(obj.fid);
            if ~ischar(line) && line == -1
                err = 'EOF';
                cnt = -1;
            elseif isempty(line)
                err = ferror(obj.fid);
                line = 0;
                cnt = -1;
            else
                cnt = length(line);
                line = regexprep(line, '\\n$', '');
                err = '';
            end
        end
    end    
end

