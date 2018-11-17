classdef SensingMatrixNrDCT < SensingMatrixSqrNrRnd & SensingMatrixDCT
    %SensingMatrixNrWH Same as SensingMatrixWH, except that the
    %input is not randomized
    
    properties
    end
    
    methods
        function obj = SensingMatrixNrDCT(varargin)
            obj.setSensingMatrixDCT(varargin{:})
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixDCT(varargin{:});
        end
        
        function n_no_clip=nNoClip(obj)
            n_no_clip = obj.nNoClip@SensingMatrixDCT();
        end
        
        function dc_val = getDC(obj,msrs)
            dc_val = obj.getDC@SensingMatrixDCT(msrs);
        end
    end
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order)
            [PL, PR] = obj.makePermutations@SensingMatrixSqrNrRnd(order);
        end
    end

end
