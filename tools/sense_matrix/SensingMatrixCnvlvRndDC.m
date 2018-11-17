classdef SensingMatrixCnvlvRndDC < SensingMatrixConcat
    % SensingMatrixCnvlvRndDC - Same as SensingMatrixCnvlvRnd but adds a
    % DC measurement.
    %  The last measurement is discarded to keep the same number of
    %  measurements.
    
    properties
    end
    
    methods
        function obj = SensingMatrixCnvlvRndDC(varargin)
            obj.setSensingMatrixCnvlvRndDC(varargin{:})
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'prmt', 'num_columns', ...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixCnvlvRndDC(varargin{:})
        end
        
    end
    
    methods (Access=protected)
        function setSensingMatrixCnvlvRndDC(obj, prmt, num_columns,...
                rnd_seed, order, rnd_type)
            if nargin < 3
                smr_args = {};
            else
                if isfield(prmt, 'N_msrs')
                    orig_prmt = prmt;
                    prmt = SensingMatrixCnvlvRnd.calcPermutSizes(prmt);
                    if  prmt.N_msrs == orig_prmt.N_msrs
                        orig_prmt.N_msrs = orig_prmt.N_msrs-1;
                    end
                    prmt = orig_prmt;
                end
                
                switch nargin
                    case 3
                        mtrx_cv = SensingMatrixCnvlvRnd(prmt, num_columns);
                    case 4
                        mtrx_cv = SensingMatrixCnvlvRnd(prmt, num_columns,...
                            rnd_seed);
                    case 5
                        mtrx_cv = SensingMatrixCnvlvRnd(prmt, num_columns,...
                            rnd_seed, order);
                    case 6
                        mtrx_cv = SensingMatrixCnvlvRnd(prmt, num_columns,...
                            rnd_seed, order, rnd_type);
                end
                mtrx_dc = SensingMatrixDC(num_columns);
                matrices = {mtrx_dc, mtrx_cv};
                smr_args = {matrices};
            end
            obj.setSensingMatrixConcat(smr_args{:});
        end
    end
    
end

