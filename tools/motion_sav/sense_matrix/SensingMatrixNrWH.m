classdef SensingMatrixNrWH < SensingMatrixSqrNrRnd & SensingMatrixWH
    %SensingMatrixNrWH Same as SensingMatrixWH, except that the
    %input is not randomized
    
    properties
    end
    
    methods
        function obj = SensingMatrixNrWH(varargin)
            obj.setSensingMatrixWH(varargin{:})
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixWH(varargin{:});
        end
        
        function n_no_clip=nNoClip(obj)
            n_no_clip = obj.nNoClip@SensingMatrixWH();
        end
        
        function dc_val = getDC(obj,msrs)
            dc_val = obj.getDC@SensingMatrixWH(msrs);
        end
    end
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order, dummy)
            [PL, PR] = obj.makePermutations@SensingMatrixSqrNrRnd(order,dummy);
        end
    end

end
