classdef SensingMatrixLclDCT < SensingMatrixSqrLclRnd & SensingMatrixDCT
    %SensingMatrixLclDCT Same as SensingMatrixDCT, except that the
    %randomization is local
    
    properties
    end
    
    methods
        function obj = SensingMatrixLclDCT(varargin)
            obj.setSensingMatrixDCT(varargin{:})
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixDCT(varargin{:});
        end
        
        function n_no_clip=nNoClip(obj)
            n_no_clip = obj.nNoClip@SensingMatrixSqrLclRnd();
        end
        
        function dc_val = getDC(obj,msrs)
            dc_val = obj.getDC@SensingMatrixSqrLclRnd(msrs);
        end
    end
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order)
            [PL,PR] = obj.makePermutations@SensingMatrixSqrLclRnd(order);
        end
    end

end

