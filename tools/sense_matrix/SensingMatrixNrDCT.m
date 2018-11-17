classdef SensingMatrixNrDCT < SensingMatrixSqrNrRnd & SensingMatrixDCT
    %SensingMatrixNrWH Same as SensingMatrixWH, except that the
    %input is not randomized
    
    properties
    end
    
    methods
        function obj = SensingMatrixNrDCT(varargin)
            obj.set(varargin{:})
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixDCT(varargin{:});
        end
        
        % Returns the sum of values of the measurement which contain DC value,
        % weighted by the ratio of the DC value to other components (in
        % terms of RMS), or 0 if there is no such measurement.
        %   Input:
        %     obj - this object
        %     msrs - the measurements vector
        function dc_val = getDC(obj,msrs)
            dc_val = obj.getDC@SensingMatrixDCT(msrs);
        end
    end
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order, opts)
            PL = obj.makeRowSlctPermutation(order, opts);
            PR = [];
        end
    end

end
