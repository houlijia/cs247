classdef SensingMatrixMLSeqDC < SensingMatrixConcat
    %MLSeqDC - Same as MLSeq but adds a DC measurement.
    %  The last measurement is discarded to keep the same number of
    %  measurements.
    
    properties
    end
    
    methods
        function obj = SensingMatrixMLSeqDC(varargin)
            obj.setSensingMatrixMLSeqDC(varargin{:})
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'prmt', 'num_columns', ...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixMLSeqDC(varargin{:})
        end
        
        function n_no_clip=nNoClip(~)
            n_no_clip = 1;
        end

    end
    
    methods (Access=protected)
        function setSensingMatrixMLSeqDC(obj, prmt, num_columns,...
                rnd_seed, order, rnd_type)
            if nargin < 3
                smr_args = {};
            else
                if isfield(prmt, 'N_msrs')
                    orig_prmt = prmt;
                    prmt = SensingMatrixMLSeq.calcPermutSizes(prmt);
                    if  prmt.N_msrs == orig_prmt.N_msrs
                        orig_prmt.N_msrs = orig_prmt.N_msrs-1;
                    end
                    prmt = orig_prmt;
                end
                
                switch nargin
                    case 3
                        mtrx_ml = SensingMatrixMLSeq(prmt, num_columns);
                    case 4
                        mtrx_ml = SensingMatrixMLSeq(prmt, num_columns,...
                            rnd_seed);
                    case 5
                        mtrx_ml = SensingMatrixMLSeq(prmt, num_columns,...
                            rnd_seed, order);
                    case 6
                        mtrx_ml = SensingMatrixMLSeq(prmt, num_columns,...
                            rnd_seed, order, rnd_type);
                end
                mtrx_dc = SensingMatrixDC(num_columns, mtrx_ml.trnspScale());
                matrices = {mtrx_dc, mtrx_ml};
                smr_args = {matrices};
            end
            obj.setSensingMatrixConcat(smr_args{:});
        end
    end
    
end

