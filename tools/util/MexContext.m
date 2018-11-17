classdef MexContext
  %MexContext takes care of creation and deletion of MexContext
  %   Detailed explanation goes here
  
  properties
    % uint8 arrary which contains the value of a C++ pointer to 
    % a C++ MexContext object
    mx_cntxt;
  end
  
  methods
    function obj = MexContext(mcntx)
      if nargin == 0
        mcntx = init_mex_context_mex();
      end
      obj.mx_cntxt = mcntx;
    end
    
    function delete(~)
      delete_mex_context_mex();
    end
  end
  
end

