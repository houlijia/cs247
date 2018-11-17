classdef CodeDestArray < CodeStoreArray & CodeDest
    % CodeDestArray is an implementation of CodeDest as a memory array
    
    properties
    end
    
    methods
      % CodeDestArray - constructor.
      function obj = CodeDestArray(incr_s)
        if nargin < 1
          incr_s = 1;
        end
        obj@CodeStoreArray(incr_s);
      end
      
    end    
end

