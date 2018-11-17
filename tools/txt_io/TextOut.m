classdef TextOut < handle
    % TextOut - abstract class for writing lines of text
    
    properties
    end
    
    methods (Abstract)
        % Write out a line, appending newline. return number of chars
        % written, including the new line, or error string if an error
        % occurred
        %   cnt - number of chars written (including new line). If error,
        %         returns an error string
        cnt = writeLine(obj,line);
    end
    
    methods
        % write a string as a sequence of lines of text
        function cnt = write(obj,str)
            lines = regexp(str, '\n','split');
            cnt = 0;
            for k=1:length(lines)
                n = obj.writeLine(lines{k});
                if ischar(n)
                    cnt = n;
                    break;
                end
                cnt = cnt + n;
            end
        end
    end
end

