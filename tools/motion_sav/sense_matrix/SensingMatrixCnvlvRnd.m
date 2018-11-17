classdef SensingMatrixCnvlvRnd < ...
        SensingMatrixSqrRnd & SensingMatrixConvolve 
    %SensingMatrixCnvlvRnd - A specialization of SensingMatrixConvovle,
    %Where the permutations and the sequence are determined randomly.
    %The sequence gets the values of {1,-1}.
    %The permutation parameters PR_start, PL_start and PL_seq need not be
    %specified: They are selected randomly. N_msrs must be always
    %specified, even when SensingMatrixConvolve it does not have to be.
    
    properties
    end
    
    methods
        % Constructor
        % Initialize the object
        % Input
        %   prmt - A permutation specification struct for
        %          SensingMatrixConvolve. However, the fields PR_start,
        %          PL_start, PL_seq need not be specified - they are
        %          created randomly.  In mode MODE_ARBT, N_msrs has to be
        %          prvided instead of PL_seq.
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
        %          created randomly.  In mode MODE_ARBT, N_msrs has to be
        %          prvided instead of PL_seq.
        %   rnd_seed - Randomization seed
        %   order - Order of square matrix
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'prmt', 'num_columns',...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixCnvlvRnd(varargin{:});
        end
        
        function makeSequence(obj)
            sq = obj.rnd_strm.randi([0,1],[obj.sqr_order, 1]);
            
            obj.setSequence(1-2*sq);
            obj.setSeqType(3,[1 -1]);
        end
        
        function len=encode(obj, code_dst, info)
            len = obj.encode@SensingMatrixSqrRnd(code_dst, info);
            if ischar(len)
                retrun
            end
            total = len;
            
            len = obj.encodeSequence(code_dst, info);
            if ischar(len)
                return;
            end
            len = total + len;
        end
        
        function len=decode(obj, code_src, info, cnt)
            n_read = obj.decode@SensingMatrixSqrRnd(code_src, info, cnt);
            len = n_read;
            if ischar(n_read)
                return;
            end
             
            n_read = obj.decodeSequence(code_src, info, cnt-len);
            if ischar(n_read)
                len = n_read;
                return;
            end
            len = len+n_read;
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          if ~obj.is_transposed
            y = y / sqrt(obj.nCols());
          else
            y = y / (sqrt(obj.nCols()) * obj.trnspScale());
          end
        end

        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          if ~obj.is_transposed
            y = y * sqrt(obj.nCols());
          else
            y = y * (sqrt(obj.nCols()) * obj.trnspScale());
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
        %          created randomly.  In mode MODE_ARBT, N_msrs has to be
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
                prmt = SensingMatrixConvolve.calcPermutSizes(prmt);
%                 fprintf('permut sizes=%s\n', int2str(prmt.PL_range));
                switch nargin
                    case 3
                        smr_args = { prmt.N_msrs, num_columns };
                    case 4
                        order = SensingMatrixConvolve.calcDefaultOrder(...
                            num_columns, prmt);
                        smr_args = { prmt.N_msrs, num_columns, ...
                            rnd_seed, order, prmt};
                    case 5
                        smr_args = { prmt.N_msrs, num_columns, ...
                            rnd_seed, order, prmt};
                    case 6
                        smr_args = { prmt.N_msrs, num_columns, ...
                            rnd_seed, order, prmt, rnd_type};
                end
            end
            obj.setSensingMatrixSqrRnd(smr_args{:});
            
            if nargin >= 4
                obj.makeSequence()
            end
        end
        
        function [PL, PR] = makePermutations(obj, order, prmt)
            if obj.unit_permut_R
                prmt.PR_start = 0;
            else
                prmt.PR_start=obj.rnd_strm.randi([0 (order-obj.nCols())]);
            end

            if obj.unit_permut_L
                prmt.PL_mode = SensingMatrixConvolve.MODE_1D;
                prmt.PL_start= 0;
            else
                switch prmt.PL_mode
                    case SensingMatrixConvolve.MODE_ARBT
                        prmt.PL_seq = obj.rnd_strm.randperm(order);
                        prmt.PL_seq = prmt.PL_seq(1:prmt.N_msrs);
                    case SensingMatrixConvolve.MODE_1D
                        prmt.PL_start=obj.rnd_strm.randi([0 (order-1)]);
                    case SensingMatrixConvolve.MODE_LCLS
                        prmt.PL_seq = obj.rnd_strm.randperm(order);
                        prmt.PL_seq = prmt.PL_seq(1:prmt.N_msrs);
                    case SensingMatrixConvolve.MODE_GLBL
                        prmt.PL_start=obj.rnd_strm.randi([0 (order-1)]);
                end
            end
            obj.permut = prmt;
            [PL,PR] = obj.makePermutations@SensingMatrixConvolve(order, prmt);
            
%             PR = SensingMatrixConvolve.calcPR(prmt, order, obj.nCols());
%             if ~any(prmt.PL_range(1:3))
%                 PL = obj.rnd_strm.randperm(order);
%             else
%                 if length(prmt.PL_range) == 4 && ~prmt.PL_range(4)
%                     PL = obj.calcRandRangePrmt(prmt, order);
%                 else
%                     prmt.PL_start=obj.rnd_strm.randi([0 (order-1)]);
%                     PL = SensingMatrixConvolve.calcPL(prmt, order);
%                 end
%             end
%             obj.permut = prmt;
        end
        
%         function PL = calcRandRangePrmt(obj, prmt, order)
%             prmt.PL_start=0;
%             prmt.PL_range(4) = 1;
%             slc = SensingMatrixConvolve.calcSlctPL(prmt);
%             prmt.PL_range(4) = 0;
%             pl = obj.rnd_strm.randperm(order);
%             %pl = pl(1:prmt.N_msrs);
%             PL = slc * ones(1, order) + ones(size(slc))*pl;
%             PL = mod(PL(:)-1, order)+1;
%             PL = unique(PL, 'stable');
% %             PL = PL(1:prmt.N_msrs);
% %             PO = (1:order)';
% %             PO(PL) = [];
% %             PL = [PL; PO];
%             
% %             sz = [prmt.PL_size(1:3) ceil(prmt.N_msrs / prod(prmt.PL_size(1:3)))];
% %             lsz = prod(sz);
% %             hblk = reshape((1:lsz)', sz);
% %             orig = hblk(1:end-prmt.PL_range(1)+1, 1:end-prmt.PL_range(2)+1,...
% %                 1:end-prmt.PL_range(3)+1, :);
% %             orig = orig(:);
% %             orig(obj.rnd_strm.randperm(length(orig))') = orig;
% %             
% %             pl = zeros(prod(prmt.PL_range(1:3)), length(orig));
% %             for k = 1:length(orig);
% %                 og = orig(k);
% %                 [v,h,t,c] = ind2sub(sz,og);
% %                 slc = hblk(v:v+prmt.PL_range(1)-1, h:h+prmt.PL_range(2)-1,...
% %                     t:t+prmt.PL_range(3)-1, c);
% %                 pl(:,k) = slc(:);
% %             end
% %             
% %             pl = unique(pl(:), 'stable');
% %             pl = mod(pl+(prmt.PL_start-1), order) + 1;
% %             pl = sort(pl(1:prmt.N_msrs));
% %             PL = (1:order)';
% %             PL(pl) = [];
% %             PL = [pl;PL];
%         end
        
        function len=encodeSequence(~, ~, ~)
            len = 0;
        end
       
        function len = decodeSequence(obj, ~, ~, ~)
           obj.makeSequence();
           len=0;
        end
        
        function len=encodePermutations(obj, code_dst, ~)
            len = code_dst.writeUInt([obj.permut.PL_mode, ...
                obj.permut.PL_range(1:3), obj.permut.PL_size(1:3), obj.permut.N_msrs]);
        end
        
       function [prmt_info, len] = decodePermutations(obj, code_src, ~, cnt)
            [vals, len] = code_src.readUInt(cnt, [1 8]);
            if ischar(vals) || (isscalar(vals) && vals == -1)
                if ischar(vals)
                    prmt_info = vals;
                else
                    prmt_info = 'EOD encountered';
                end
                return;
            end
            vals = double(vals);
            prmt_info = struct('PL_mode', vals(1), 'PL_range', vals(2:4),... 
                'PL_size', vals(5:7), 'N_msrs', vals(8));
            prmt_info = obj.calcPermutSizes(prmt_info);
       end
        
    end 
    
%     methods (Static)
%         
%         function prmt = calcPermutSizes(prmt)
%             if isfield(prmt, 'min_order')
%                 return;
%             end
%             if length(prmt.PL_range) == 4 && ~prmt.PL_range(4)
%                 prmt.min_order =0;
%                 return;
%             end
%             
%             prmt = SensingMatrixConvolve.calcPermutSizes(prmt);
%         end
%         
%     end        
    
end

