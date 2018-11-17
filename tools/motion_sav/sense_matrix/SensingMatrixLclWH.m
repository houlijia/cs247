classdef SensingMatrixLclWH < SensingMatrixSqrLclRnd & SensingMatrixWH
    %SensingMatrixLclWH Same as SensingMatrixWH, except that the
    %randomization is local
    
    properties
    end
    
    methods
        function obj = SensingMatrixLclWH(varargin)
            obj.setSensingMatrixWH(varargin{:})
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixWH(varargin{:});
        end
        
        function n_no_clip=nNoClip(obj)
            n_no_clip = obj.nNoClip@SensingMatrixSqrLclRnd();
        end
        
        function dc_val = getDC(obj,msrs)
            dc_val = obj.getDC@SensingMatrixSqrLclRnd(msrs);
        end
    end
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order, dummy)
            [PL, PR] = obj.makePermutations@SensingMatrixSqrLclRnd(order,dummy);
        end
    end

end

