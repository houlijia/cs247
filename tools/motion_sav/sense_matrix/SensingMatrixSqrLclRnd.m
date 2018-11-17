classdef SensingMatrixSqrLclRnd <  SensingMatrixSqrLcl & SensingMatrixSqrRnd
    %SensingMatrixSqrLclRnd is the same as SensingMatrixSqrRnd, except that 
        %instead of SensingMatrixSqr we use SensingMatrixSqrLcl
    
    properties
    end
    
    methods
        %Constructor
        function obj = SensingMatrixSqrLclRnd(varargin)
            obj.set(varargin{:});
        end
        
        function set(obj, varargin)
            obj.set@SensingMatrixSqrRnd(varargin{:})
        end

        function y=doMultVec(obj, varargin)
            y = obj.doMultVec@SensingMatrixSqrLcl(varargin{:});
        end
        
        function y=doMultTrnspVec(obj,varargin)
            y = obj.doMultTrnspVec@SensingMatrixSqrLcl(varargin{:});
        end
    end    
    
    methods(Access=protected)
        function [PL, PR] = makePermutations(obj, order, ~)
            if obj.unit_permut_L
                PL = 1:order;
            else
                PL = obj.rnd_strm.randperm(order);
            end
            if obj.unit_permut_R
                PR = 1:order;
            else
                PR = obj.makeLclPR(order);
            end
        end
        
        function PR = makeLclPR(obj,order)
            PR = (-1) .^ obj.rnd_strm.randi([0,1],[1,order]);
        end
    end

end

