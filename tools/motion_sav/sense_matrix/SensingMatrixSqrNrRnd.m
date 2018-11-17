classdef SensingMatrixSqrNrRnd <  SensingMatrixSqrRnd
    %SensingMatrixSqrNrRnd is the same as SensingMatrixSqrRnd, except that 
    %the input is not reordered (PR is the unit permutation).
    
    properties
    end
    
    methods
        %Constructor
        function obj = SensingMatrixSqrNrRnd(varargin)
            obj.set(varargin{:});
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixNrRnd(varargin{:})
        end

    end    
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order, ~)
            if obj.unit_permut_L
                PL = 1:order;
            else
                PL = obj.rnd_strm.randperm(order);
            end
            PR = 1:order;
        end
        
    end

end
