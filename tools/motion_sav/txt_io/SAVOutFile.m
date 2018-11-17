classdef SAVOutFile < TextOutFile & SAVOut 
    % SAVOut where the output is into a file
    
    properties
    end
    
    methods
        % Constructor. fd is the argument for the constructor of superclass
        % TextOutFile. vblkr, fps are the arguments to the
        % constructor of superclass SAVOut.
        function obj = SAVOutFile(fd, vblkr)
            obj = obj@TextOutFile(fd);
            obj = obj@SAVOut(vblkr);
       end
    end
    
end