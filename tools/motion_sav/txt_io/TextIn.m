classdef TextIn < handle
    % TextIn - abstract class for reading lines of text
    
    properties
    end
    
    methods (Abstract)
        % Reads a line of text, if err is not empty it is the error string.
        % cnt is the number of chars read, including new line if any. line
        % is returned without the newline.
        [line, err, cnt] = read(obj)
    end
    
end

