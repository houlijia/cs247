classdef CSVOutFile < TextOutFile & CSVOut 
    % CSVOut where the output is into a file
    
    properties
    end
    
    methods
        % Constructor. fd is the argument for the constructor of superclass
        % TextOutFile. flds, nm, nrec (optional) are the arguments to the
        % constructor of superclass CSVOut.
        function obj = CSVOutFile(fd, flds, nm, nrec)
            if nargin < 4
                nrec = inf;
            end
            obj = obj@TextOutFile(fd);
            obj = obj@CSVOut(flds, nm, nrec);
       end
    end
    
end

