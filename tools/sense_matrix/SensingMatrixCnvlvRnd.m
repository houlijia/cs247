classdef SensingMatrixCnvlvRnd < ...
        SensingMatrixSqrRnd & SensingMatrixConvolve 
    %SensingMatrixCnvlvRnd - A specialization of SensingMatrixConvovle,
    %Which implments Romberg's Random Convolution sensing matrix.
    %The permutation parameters PR_start, PL_start and PL_seq need not be
    %specified: They are selected randomly. N_msrs must be always
    %specified, even when in SensingMatrixConvolve it does not have to be.
    
    properties
    end
    
    methods
        % Constructor
        % Initialize the object
        % Input
        %   prmt - A permutation specification struct for
        %          SensingMatrixConvolve. However, the fields PR_start,
        %          PL_start, PL_seq need not be specified - they are
        %          created randomly.  In mode SLCT_MODE_ARBT, N_msrs has to be
        %          provided instead of PL_seq.
        %   num_columns
        %   rnd_seed - Randomization seed
        %   order - Order of square matrix        
        function obj = SensingMatrixCnvlvRnd(varargin)
            obj.setSensingMatrixCnvlvRnd(varargin{:})
        end
        
        % Initialize the object
        % Input
        %   obj - this object
        %   prmt - A permutation specification struct for
        %          SensingMatrixConvolve. However, the fields PR_start,
        %          PL_start, PL_seq need not be specified - they are
        %          created randomly.  In mode SLCT_MODE_ARBT, N_msrs has to be
        %          prvided instead of PL_seq.
        %   rnd_seed - Randomization seed
        %   order - Order of square matrix
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'prmt', 'num_columns',...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixCnvlvRnd(varargin{:});
        end
        
        function makeSequence(obj)
          fseq = zeros(1+obj.sqr_order/2,1);
          fseq([1,1+obj.sqr_order/2]) = double(obj.rnd_strm.randi([0,1],[2, 1]));
          fseq(2:end-1) = 2*rand((obj.sqr_order/2 - 1),1);
          fseq = exp(1i*pi*fseq);
          fseq = [fseq ; conj(fseq(end-1:-1:2))];
          sq = sqrt(obj.toCPUFloat(obj.sqr_order)) * real(ifft(fseq));
          obj.setSequence(sq);
          obj.setSeqType(0);
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, then each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          y = y / sqrt(obj.toFloat(obj.nCols()));
        end

        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          y = y * sqrt(obj.toFloat(obj.nCols()));
        end
      
        % Returns the sum of values of the measurement which contain DC value,
        % weighted by the ratio of the DC value to other components (in
        % terms of RMS), or 0 if there is no such measurement.
        %   Input:
        %     obj - this object
        %     msrs - the measurements vector
        function dc_val = getDC(~,~)
            dc_val = 0;
        end
        
        function chkOrtho(obj)
          if obj.sqr_order >= obj.n_rows
            obj.setOrtho_row(true);
          end
        end
          
    end
    
    methods (Access=protected)
        % Initialize the object
        % Input
        %   obj - this object
        %   prmt - A permutation specification struct for
        %          SensingMatrixConvolve. However, the fields PR_start,
        %          PL_start, PL_seq need not be specified - they are
        %          created randomly.  In mode SLCT_MODE_ARBT, N_msrs has to be
        %          prvided instead of PL_seq.
        %   rnd_seed - Randomization seed
        %   order - Order of square matrix
        %   rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        function setSensingMatrixCnvlvRnd(obj, prmt, num_columns, ...
            rnd_seed, order, rnd_type)
          if nargin < 3
            smr_args = {};
          else
            prmt = obj.calcPermutSizes(prmt);
            % fprintf('permut sizes=%s\n', int2str(prmt.PL_range));
            if nargin < 6
              if nargin < 5
                order = obj.defaultOrder(prmt.N_msrs, num_columns, prmt);
                if nargin < 4
                  rnd_seed = SensingMatrixRnd.default_seed;
                end
              end
              smr_args = { prmt.N_msrs, num_columns, ...
                rnd_seed, prmt, order};
            else
              smr_args = { prmt.N_msrs, num_columns, ...
                rnd_seed, prmt, order, rnd_type};
            end
          end
          obj.setSensingMatrixSqrRnd(smr_args{:});
          
          if nargin >= 3
            obj.makeSequence();
            obj.chkOrtho();
          end
        end
        
        function ord = defaultOrder(obj, ~, num_columns, prmt_info)
          if nargin < 4
            prmt_info = obj.permut;
          else
            if ~isfield(prmt_info, 'N_msrs')
              prmt_info.N_msrs = num_columns;
            end
            if ~isfield(prmt_info, 'PL_mode')
              prmt_info.PL_mode = obj.permut.PL_mode;
            end
          end
          
            ord = SensingMatrixCnvlvRnd.calcDefaultOrder(num_columns,prmt_info);
        end
        
        function [PL, PR] = makePermutations(obj, order, opts)
          prmt = opts.prmt;
          prmt.PR_start = 0;
          
          switch prmt.PL_mode
            case SensingMatrixConvolve.SLCT_MODE_NONE
              prmt.PL_seq = [];
            case {SensingMatrixConvolve.SLCT_MODE_ARBT, SensingMatrixConvolve.SLCT_MODE_LCLS}
              prmt.PL_seq = obj.rnd_strm.randperm(order, prmt.N_msrs);
            case {SensingMatrixConvolve.SLCT_MODE_1D, SensingMatrixConvolve.SLCT_MODE_GLBL}
              prmt.PL_start=double(obj.rnd_strm.randi([0 (order-1)]));
          end
          obj.permut = prmt;
          [PL,PR] = obj.makePermutations@SensingMatrixConvolve(order, opts);
        end
        
        % this approximation is based on the fact that the entries of A'A are
        % N in the diagonal and have an expected values of 0 outside it:
        function y = compNorm(obj)
          y = sqrt(obj.toCPUFloat(obj.sqr_order));
        end
        
        function setCastIndex(obj)
          obj.setCastIndex@SensingMatrixConvolve();
        end
    end
    
    methods (Static)        
        function order = calcDefaultOrder(num_columns, prmt)
            if ~isfield(prmt, 'min_order')
                prmt = SensingMatrixCnvlvRnd.calcPermutSizes(prmt);
            end
            order = SensingMatrixConvolve.calcDefaultOrder(num_columns, prmt);
            order = pow2(nextpow2(double(order)));
        end
    end
    
end

