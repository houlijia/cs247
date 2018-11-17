classdef SensingMatrixConvolve < SensingMatrixSqr
    % SensingMatrixConvolve - A sensing matrix based on convolution.
    % The matrix is a circular matrix. The (i,j) entry is seq(i-j+1).
    % Each row is a shifted version of the reversed sequence seq:
    %
    %   seq(1)   seq(N)  seq(N-1)... seq(2)
    %   seq(2)   seq(1)  seq(N)...   seq(3)
    %    ....
    %   seq(N)   seq(N-1)  ....      seq(1)
    % where N is sqr_order.
    % 
    % The permutations are of a special form. The right permutation is a
    % shift and has the form
    %   PR = [pr_shift, pr_shift+1, ... sqr_order, 1,... pr_shift-1].
    % The left permutation has the form
    % { s0 + i + V*(j + H*(k + T*l)) | 0<=i<v, 0<=j<h, 0<=k<t, 0<=l<c}
    % Where V,H,T correspond to block dimensions (Horizaontal, vertical,
    % temporal), 0<v<=V, 1<h<=H, 1<t<=T, and c is represents number of
    % whole blocks.
    
    properties (Constant)  % See below for meaning
        MODE_ARBT = 1;
        MODE_1D = 2;
        MODE_LCLS = 3;
        MODE_GLBL = 4;
    end
    
    properties
        seq=[];  % The sequence to convolve with
        
        % Permutations specification for the superclass SensingMatrixSqr.
        % It is provided during initialization.  Some of the fields may be
        % modified or created during the initialization. The fields are:
        %   min_order - minimum order of the square matrix which is needed. 
        %               This field is computed by calcPermutSizes(), so its
        %               presence indicates that additional computation is not
        %               necessary.
        %  Right permutation definition fields:
        %   PR_start - The right permutatin is a shift. PR_start
        %       specifies the start of the shift. Thus
        %       PR = [pr_shift, pr_shift+1, ... sqr_order, 1,...
        %            pr_shift-1];
        %  Left permutation definition fields
        %   PL_mode - mode of generation of of the left permutation. In the
        %             following (V,H,T) are the block dimensions.
        %             PL_mode can get the following values (constants of
        %             this class).
        %     MODE_ARBT - arbitratry sequence provided by the field PL_seq.
        %                 In this case N_msrs is the length of that sequence.
        %                 Fields required: PL_seq.
        %     MODE_1D   - A contiguous sequence starting at PL_start,
        %                 wrapping around sqr_order. 
        %                 Fields required: PL_start, N_msrs.
        %     MODE_LCLS - Make sure to have measurements with specified
        %                 offsets.  Let [v,h,t] be the first 3 fields in
        %                 PL_range and let 
        %                    L(s)={s+i+V*(j+H*k) | 0<=i<v, 0<=j<h, 0<=k<t}
        %                 Consider the sequence obtained by concatenating
        %                 L(s(1)),L(s(2))),... where s(1),s(2) are given by
        %                 PL_seq.  Remove all repeated entries from this
        %                 sequence and truncate it after N_msrs entries.
        %                 This is the required sequence.
        %                 Fields required: PL_range, PL_seq, PL_size, N_msrs.
        %     MODE_GLBL - Get a maximal number of measurements with
        %                 specified offsets. Let [v,h,t] be the first 3 fields
        %                 in PL_range.  These numbers are multiplied
        %                 proportinally to be as close as possible to
        %                 (V,H,T).  Then we create as many copies of that
        %                 sequence as needed to get as close to N_msrs, by
        %                 shifting by VHT. Finally, we add PL_start.
        %                 Fields required: PL_start, PL_range, PL_size, N_msrs.
        %   PL_seq - needed only in mode MODE_ARBT. Is either a numerical
        %            sequence to be used as PL or a pointer to a function
        %            which, when called provides the
        %   PL_start - a starting point for left permutation (s0 in the
        %              above explanation).
        %   PL_range - the values of [v h t c] above.  If not all the
        %             values are non-negative, the permutation is set to be
        %             a contiguous segment of the sequence, starting at s0.
        %             In the subclass SensingMatrixCnvlvRnd, if ALL the
        %             values of PL_range are 0, a completely random
        %             permutation is generated.
        %   PL_size - the values of [V H T] above (a 4th entry may be included
        %             and is ignored.
        %   N_msrs - Optional. no. of measurements (n_rows of the matrix).  
        %            Depending on the mode, this may be computed or modified
        %            during initializaiton. 
        %            If present, calcPermutSizes() proportionately  
        %            increased or decreased PL_range so that its product is
        %            close from below to N_msrs.  If N_msrs is not present,
        %            calcPermutSizes() computes it as the product of PL_range.
        %            Note that if N_msrs is present, PL_size may include
        %            only 3 entries (v,h,t).
        permut=struct('PR_start',1,'PL_start',1,'PL_size',[]);
    end
    
    properties (Access=protected)
        seq_type=-1;  % Sequence type: -1=unknown 0=real, 1= integer, 
        % 2= signed integer, 3=binary integer
        seq_vals;  % If the sequence is binary, these are the values it gets.
        
        seq_sum_sqr=-1;
        fft_order;
        fft_seq;
        fft_trnsp_seq;
        
        % use_fft can be 0,1, or 2.
        %   0 - do not use FFT - use Matlab conv instead.
        %   1 - use FFT instead of conv
        %   2 - use FFT and check it against conv
        use_fft = 1;
    end
    
    methods
        % Constructor
        %   Input (all input values are optional
        %     prmt - A struct of the same structure as obj.permut
        %     num_columns - number of columns
        %     sqnc - the sequence used to create the matrix. sqr_order is
        %            the length of sqnc
        %         
        %            
        function obj = SensingMatrixConvolve(varargin)
            obj.setSensingMatrixConvolve(varargin{:})
        end
        
        % Set values. Same signature as constructor
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'prmt', 'num_columns', 'sqnc'});
            obj.setSensingMatrixConvolve(varargin{:});
        end
        
        function aux_info = setSequence(obj, sqnc, aux_info)
            if iscolumn(sqnc)
                obj.seq = sqnc;
            else
                obj.seq = sqnc';
            end
            
            if obj.use_fft
                if nargin >= 3
                    obj.fft_seq = aux_info.fft_seq;
                    obj.fft_trnsp_seq = aux_info.fft_trnsp_seq;
                    obj.fft_order = aux_info.fft_order;
                else
                    N=length(sqnc);
                    obj.fft_order = pow2(nextpow2(N*2));
                    fft_len = obj.fft_order/2 + 1;
                    SQ = fft([obj.seq(2:end); obj.seq;...
                        zeros(obj.fft_order-(2*N-1),1)]);
                    obj.fft_seq = SQ(1:fft_len);
                    SQ = fft([obj.seq(end:-1:1);obj.seq(end:-1:2);...
                        zeros(obj.fft_order-(2*N-1),1)]);
                    obj.fft_trnsp_seq = SQ(1:fft_len);
                    aux_info = struct(...
                        'fft_order', obj.fft_order,...
                        'fft_seq', obj.fft_seq,...
                        'fft_trnsp_seq', obj.fft_trnsp_seq);
                end
            else
                aux_info = struct();
            end
        end
        
        function [sq_type, sq_vals] = calcSeqType(obj, sqnc)
            if nargin < 2
                sqnc = obj.seq;
            end
            sq_vals=[];
            if all(fix(sqnc)==sqnc)
                sqv = unique(sqnc, 'stable');
                if length(sqv) <= 2
                    sq_type = 3;
                    sq_vals = sqv;
                elseif all(sqv >= 0)
                    sq_type = 2;
                else
                    sq_type = 1;
                end
            else
                sq_type = 0;  % real
            end
        end
        
        function setSeqType(obj, sq_type, sq_vals)
            obj.seq_type = sq_type;
            if sq_type == 3
                obj.seq_vals = sq_vals;
            end
        end
        
        function len=encode(obj, code_dst, info)
            % encode Basic SensingMatrix info
            len = obj.encode@SensingMatrixSqr(code_dst, info);
            if ischar(len)
                return;
            end
            cnt = len;
            
            % Encode the sequence
            len = obj.encodeSequence(code_dst, info);
            if ischar(len)
                return;
            end
            cnt = len+cnt;
            
            %encode the permutations
            len = obj.encodePermutations(code_dst, info);
            if ischar(len)
                return;
            end
            len = len+cnt;
        end
        
        function len=decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            % Decode Basic SensingMatrix info
            n_read = obj.decode@SensingMatrixSqr(code_src, info, cnt);
            len = n_read;
            if ischar(n_read)
                return;
            end
            
            % Decode the sequence
            n_read = obj.decodeSequence(code_src, info, cnt-len);
            if ischar(n_read)
                len = n_read;
                return
            end
            len = len+n_read;
            
            [prmt, n_read] = obj.decodePermutations(code_src, info, cnt-len);
            if ischar(prmt)
                len = prmt;
                return
            end
            len = len+n_read;
            
            [pl,pr] = obj.makePermutations(obj.sqr_order, prmt);
            obj.setPermutations(obj.sqr_order,pl,pr);
        end
        
        %Multiply the vector x of size (sqr_order,1) by S
        function y=multSqr(obj,x)
            if ~obj.use_fft
                y = conv(x, [obj.seq(2:end); obj.seq], 'same');
            else
                y = obj.calcMultSqr(x,obj.fft_seq);
                if obj.use_fft == 2
                    yc = conv(x, [obj.seq(2:end); obj.seq], 'same');
                    if ~isequal(y,yc)
                        err = norm(y-yc,inf);
                        fprintf('SensingMatrixConvolve.multSqr diff: %g (%g)\n',...
                           err, err/(norm(yc,inf)+1e-16));
                    end
                end
            end
        end
        
        %Multiply the vector x of size (sqr_order,1) by S'
        function y=multTrnspSqr(obj,x)
            if ~obj.use_fft
                y= conv(x, [obj.seq(end:-1:1);obj.seq(end:-1:2)], 'same');
            else
                y = obj.calcMultSqr(x,obj.fft_trnsp_seq);
                if obj.use_fft == 2
                    yc= conv(x, [obj.seq(end:-1:1);obj.seq(end:-1:2)], 'same');
                    if ~isequal(y,yc)
                        err = norm(y-yc,inf);
                        fprintf('SensingMatrixConvolve.multTrnspSqr diff: %g (%g)\n',...
                            err, err/(norm(yc,inf)+1e-16+norm(yc,inf)));
                    end
                end
            end
            y = y * obj.trnspScale();
        end
        
        function ord = defaultOrder(~, ~, num_columns, prmt_info)
            if nargin < 4
                prmt_info = obj.permut;
            end
            ord = SensingMatrixConvolve.calcDefaultOrder(num_columns,prmt_info);
        end
        
        % Sometimes multTrnspVec may multiply the output y by a scaling
        % factor.  This function returns the scaling factor.
        function y = trnspScale(~)
            y = 1;
        end

        % A rather crude approximation for the norm of A'A.
        function y = normAtA(obj)
            if obj.seq_sum_sqr < 0;
                obj.seq_sum_sqr = double(dot(obj.seq, obj.seq));
            end
            y = obj.seq_sum_sqr;
        end
        
        % Get an array of measurements which correspond to specific offsets.
        %   Input
        %     obj: This object
        %     ofsts: a vector of offsets of length lofst.
        %     msrs: The measurements vector
        %     inp_list: Can be n array which cotains measurement indices of the
        %               measurements to use in msrs, or the string 'all'
        %     params: Optional struct of parameters which may be of use to some
        %             subclasses. Possible arguments include:
        %               nrm - norm with which comparison is to be done.
        %               ofsts_list - a column vector of indices of columns of
        %                           ofst_msrs (see below)
        %               nghbr_list - an array of indices of columns of
        %                            ofst_msrs. The number of rows is the
        %                            same as length(params.ofsts_list
        %  Output
        %     ofst_msrs: An array of size [m, lofts]. The i-th column
        %        contains the measurements (or modified measurements)
        %        corresponding to the offsets ofsts(i).
        %
        %  Note: If params is present and has the fields ofsts_list
        %        and nghbr_list, then after computing ofst_msrs it is
        %        modified by a call
        %    ofst_msrs = obj.getEdgeMsrmnts(ofst_msrs, 
        %                                    params.ofsts_list,
        %                                    params.nghbr_list);
        %        or something functionally equivalent. 
        function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts, msrs, inp_list, ...
            params)
          if ~isnumeric(inp_list)
            inp_list = 1:obj.nRows();
          end
          if ~isempty(obj.zeroed_rows)
            [~,zr,~] = intersect(inp_list, obj.zeroed_rows);
            inp_list(zr) = [];
          end
          
          ofst_indcs = obj.getOffsetIndices(ofsts, inp_list);
          
          % This complicated notation, instead of the simple
          %                ofst_msrs = msrs(msrs_indices);
          % is to overcome a Matlab idiosyncracy when size(msrs_indices,1)==1
          ofst_msrs = reshape(msrs(ofst_indcs(:)), size(ofst_indcs));
          
          if nargin >= 5 && ...
              isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
            ofst_msrs  = obj.getEdgeMsrmnts(ofst_msrs, params.ofsts_list,...
              params.nghbr_list);
          end
        end
        
    end   % methods
    
    methods (Access=protected)
        % Initialize the matrix
        %   Input (all input values are optional
        %     prmt - A struct of the same structure as obj.permut.
        %     Important fields are:
        %     num_columns - number of columns
        %     sqnc - the sequence used to create the matrix. sqr_order is
        %            the length of sqnc
        %         
        function setSensingMatrixConvolve(obj, prmt, num_columns, sqnc)
            if nargin < 3
                smr_args = {};
            else
                prmt = SensingMatrixConvolve.calcPermutSizes(prmt);
                if nargin < 4
                    smr_args = { prmt.N_msrs, num_columns };
                else
                    smr_args = { prmt.N_msrs,...
                        num_columns, length(sqnc),...
                        SensingMatrixConvolve.calcPL(prmt, length(sqnc)), ...
                        SensingMatrixConvolve.calcPR(prmt, length(sqnc),...
                        num_columns)};
                end
            end
            obj.setSensingMatrixSqr(smr_args{:});
            
            if nargin >= 2
                obj.permut = prmt;
                if nargin >= 3
                    obj.setSequence(sqnc);
                    [sq_type, sq_vals] = obj.calcSeqType(sqnc);
                    obj.setSeqType(sq_type, sq_vals);
                end
            end
        end
        
        function len=encodeSequence(obj, code_dst, ~)
            % Write sequence type
            len = code_dst.writeUInt(obj.seq_type);
            if ischar(len)
                return;
            end
            total = len;

            % Write values if necessary
            if obj.type == 3
                len = code_dst.writeSInt(obj.seq_vals);
                if ischar(len)
                    return;
                end
                total = total+len;
            end
            
            switch (obj.seq_type)
                case 0
                    len = code_dst.writeNumber(obj.seq);
                case 1
                    len = code_dst.writeSInt(obj.seq);
                case 2
                    len = code_dst.writeUInt(obj.seq);
                case 3
                    len = code_dst.writeBits(~(obj.seq==obj.seq_vals(1)));
            end
            if ischar(cnt)
                return;
            end
            len = len+total;
        end
       
        function len = decodeSequence(obj, code_src, ~, cnt)
            %Decode  order and type
            [typ, n_read] = code_src.readUInt(cnt);
            if ischar(typ) || (isscalar(typ) && typ == -1)
                if ischar(typ)
                    len = typ;
                else
                    len = 'Unexpected end of data';
                end
                return;
            end
            len = n_read;
            
            % read sequence values if necessary
            if typ == 3
                [sqvl, n_read] = code_src.readSInt(cnt-len, [1 2]);
                if ischar(sqvl)
                    len = sqvl;
                    return;
                end
                len = len + n_read;
            end
            
            switch typ
                case 0
                    [sq, n_read] = code_src.readNumber(cnt-len, [obj.sqr_order,1]);
                case 1
                    [sq, n_read] = code_src.readSInt(cnt-len, [obj.sqr_order,1]);
                case 2
                    [sq, n_read] = code_src.readUInt(cnt-len, [obj.sqr_order,1]);
                case 3
                    [sq, n_read] = code_src.readBits(cnt-len);
            end
            if ischar(sq)
                len = sq;
                return;
            end
            len = len + n_read;

           if typ == 3
                sqv = sqvl(1)*ones(size(sq));
                sqv(sq ~= 0) = sqvl(2);
                sq = sqv;
                obj.seq_vals = sqvl;
           end
            
           obj.seq_type = typ;
           obj.setSequence(sq);
        end
       
        function [PL, PR] = makePermutations(obj, order, prmt)
            PL = SensingMatrixConvolve.calcPL(prmt, order);
            PR = SensingMatrixConvolve.calcPR(prmt, order, obj.n_cols);
            obj.permut = prmt;
        end
        
       function len=encodePermutations(obj, code_dst, ~)
           len = code_dst.writeUInt([obj.permut.PR_start, obj.permut.PL_mode]);
           if ischar(len)
               return;
           end
           total = len;
           
           switch obj.permut.PL_mode
               case SensingMatrixConvolve.MODE_ARBT
                   len = obj.encodePL_seq(code_dst);
               case SensingMatrixConvolve.MODE_1D
                   len = code_dst.writeUInt(obj.permut.PL_start);
               case SensingMatrixConvolve.MODE_LCLS
                   len = code_dst.writeUInt(PL_range(1:3));
                   if ischar(len)
                       return;
                   end
                   total = total+len;
                   len = obj.encodePL_seq(code_dst);
               case SensingMatrixConvolve.MODE_GLBL
                   len = code_dst.writeUInt(PL_range(1:3));
           end
           if ischar(len)
               return;
           end
           len = len + total;
                   end
        
        function [prmt_info, len] = decodePermutations(~, code_src, ~, cnt)
            [vals, len] = code_src.readUInt(cnt, [1 2]);
            if ischar(vals) || (isscalar(vals) && vals == -1)
                if ischar(vals)
                    prmt_info = vals;
                else
                    prmt_info = 'EOD encountered';
                end
                return;
            end
            cnt = cnt - len;
            total = len;
            vals = double(vals);
            prmt_info = struct('PR_start',vals(1), 'PL_mode', vals(2));
            
            switch prmt_info.PL_mode
               case SensingMatrixConvolve.MODE_ARBT
                   [plseq, len] = obj.decodePL_seq(code_src,cnt);
                   if ischar(plseq)
                       prmt_info = plseq;
                       return;
                   end
                   len = len + total;
                   prmt_info.PL_seq = plseq;
                case SensingMatrixConvolve.MODE_1D
                    [pl_start, len] = code_src.readUInt(cnt);
                    if ischar(pl_start)
                        prmt_info = pl_start;
                        return;
                    end
                    len = len+total;
                    prmt_info.PL_start = pr_start;
                case SensingMatrixConvolve.MODE_LCLS
                    [pl_range, len] = code_src.readUInt(cnt, [1 3]);
                    if ischar(pl_range)
                        prmt_info = pl_range;
                        return;
                    end
                    cnt = cnt-len;
                    total = total + len;
                    [plseq, len] = obj.decodePL_seq(code_src,cnt);
                    if ischar(plseq)
                        prmt_info = plseq;
                        return;
                    end
                    len = len + total;
                    prmt_info.PL_range = pl_range;
                    prmt_info.PL_seq = plseq;
                case SensingMatrixConvolve.MODE_GLBL
                    [pl_range, len] = code_src.readUInt(cnt, [1 3]);
                    if ischar(pl_range)
                        prmt_info = pl_range;
                        return;
                    end
                    len = total + len;
                    prmt_info.PL_range = pl_range;
            end
            
            prmt_info.N_msrs = obj.n_rows;
           
%             if ischar(vals) || (isscalar(vals) && vals == -1)
%                 if ischar(vals)
%                     prmt_info = vals;
%                 else
%                     prmt_info = 'EOD encountered';
%                 end
%                 return;
%             end
%             prmt_info = struct('PR_start', vals(1), 'PL_start', vals(2),...
%                 'PL_range', vals(3:6), 'PL_size', vals(7:9));
%             
%             prmt_info = SensingMatrixConvolve.calcPermutSizes(prmt_info);
        end
         
        % Get a list of measurments indices with specified offsets
        %   Input:
        %     obj: This object
        %     ofsts: a vector of offsets of length lofst.
        %     inp_list: A list of input measurement numbers to use
        %   Output
        %     indcs: An array with rows of lengths lofst. Each row
        %           contains indices in inp_list such that if i is
        %           the n-th index in the row and j is the k-th index, then
        %           obj.IPL(inp_list(j))-obj.IPL(inp_list(i)) = 
        %              ofsts(k) - ofst(n)  mod(obj.sqr_order) 
        function indcs = getOffsetIndices(obj, ofsts, inp_list)
            lofst = length(ofsts);
            
            if ~isempty(obj.zeroed_rows)
                [~,zr,~] = intersect(inp_list, obj.zeroed_rows);
                inp_list(zr) = [];
            end
                
            % ipl is the list of the square matrix rows which are selected.
            if obj.is_transposed
                ipl = obj.PR;
            else
                ipl = obj.IPL;
            end
            
            % Start assuming that all indices will fit
            indcs = zeros(length(inp_list), lofst);
            indcs(:,1) = inp_list(:);
            sq_indcs = ipl(inp_list(:));
            ofsts = ofsts(2:end)-ofsts(1);
            
            % for each offset, check for each measurement in the first
            % column if the offseted measurement exists.  Delete rows where
            % it does not exist and put the ones that exists in their place
            % in the k+1 column
            for k=2:lofst
                % ofst is a vector with the values in ipl shifted by
                % ofsts(k).
                ofst = mod((sq_indcs + (ofsts(k-1)-1)), obj.sqr_order)+1;
                
                % in the result, for each k,
                %      ipl(ind_ipl(k)) = ofst(ind_ofst(k)
                % We rely on the fact that ipl and ofst have unique,
                % non-zero values.
                [~,ind_inp, ind_ofst] = intersect(sq_indcs,ofst);
                
                % Set the matches to the measurements numbers
                indcs(ind_ofst,k) = ind_inp;
            end
            
            % delete rows which contain zeros
            indcs(~prod(double(logical(indcs(:,2:end))),2),:) = [];
        end
        
    end
    
    methods (Access=private)
        %Multiply the vector x of size (sqr_order,1) by S
        function y=calcMultSqr(obj,x,SQ)
            X = fft([x;zeros(obj.fft_order-length(x),1)]);
            Y = SQ .* X(1:length(SQ));
            y = real(ifft([Y; conj(Y(end-1:-1:2))]));
            y = y(obj.sqr_order:2*obj.sqr_order-1);
        end
    end
    
    methods(Static)
        function num_rows = calcNRows(prmt)
            num_rows = prod(prmt.PL_range);
        end
        
        % Calculate a set of offset for a range (3 dim) and a given block
        % size sz.
        function pl = calcSlctPL(range, sz)
            pl = (0:(range(end)-1))';
            for d = (length(range)-1):-1:1
                pl = sz(d)*ones(range(d),1)*pl' + ...
                    (0:range(d)-1)'*ones(1,length(pl));
                pl = pl(:);
            end
        end
        
        function pl_prmt = calcPL(prmt, sqln)
            switch prmt.PL_mode
                case SensingMatrixConvolve.MODE_ARBT
                    plseq = double(prmt.PL_seq);
                    if ~iscolumn(plseq)
                        plseq = plseq';
                    end
                    prmt.N_msrs = length(plseq);
                case SensingMatrixConvolve.MODE_1D
                    plseq = prmt.PL_start + (1:prmt.N_msrs)';
                    plseq = 1 + mod(plseq(:)-1,sqln);
                case SensingMatrixConvolve.MODE_LCLS
                    plseq = SensingMatrixConvolve.calcSlctPL(...
                        prmt.PL_range(1:3), prmt.PL_size);
                        
                    plseq = plseq * ones(1,length(prmt.PL_seq)) + ...
                        ones(size(plseq))*prmt.PL_seq;
                    plseq = 1 + mod(plseq(:)-1,sqln);
                    plseq = unique(plseq, 'stable');
                    plseq = plseq(1:prmt.N_msrs);
                case SensingMatrixConvolve.MODE_GLBL
                    prmt.PL_range = SensingMatrixConvolve.findPermutSizes(...
                        prmt.PL_range(1:3), prmt.PL_size, prmt.N_msrs);
                    plseq = SensingMatrixConvolve.calcSlctPL(...
                        prmt.PL_range, prmt.PL_size);
                    plseq = plseq + prmt.PL_start;
                    plseq = 1 + mod(plseq(:)-1,sqln);
            end
            
            pl_ext = (1:sqln)';
            pl_ext(plseq) = [];
            pl_ext = [plseq;pl_ext];
            
            %invert the permutation
            pl_prmt = zeros(sqln,1);
            pl_prmt(pl_ext) = (1:sqln)';
        end
        
        function pr_prmt = calcPR(prmt, sqln, ~)
            pr_prmt = prmt.PR_start + (1:sqln);
            pr_prmt = 1 + mod(pr_prmt(:)-1,sqln);
        end
        
        function order = calcDefaultOrder(num_columns, prmt_info)
            order = max(num_columns, prmt_info.min_order);
        end
        
        function range = findPermutSizes(base_rng, cube_size, n_msrs)
            if ~all(base_rng)
                range = [n_msrs, 1, 1, 1];
                return
            end
            c = ceil(n_msrs/prod(cube_size));
            ref_size = n_msrs/c;
            f = (n_msrs/(c*prod(base_rng)))^(1/3);
            base_rng = floor(f * base_rng(1:3));
            for k=1:3
                if base_rng(k) > cube_size(k)
                    base_rng = floor(base_rng * sqrt(base_rng(k)/cube_size(k)));
                    base_rng(k) = cube_size(k);
                end
            end
            base_rng = min(base_rng, cube_size);
            base_ttl = prod(base_rng);
            
            while true
                range = base_rng;
                ttl = base_ttl;
                for k=1:6
                    rn = base_rng;
                    switch k
                        case {1,2,3}
                            rn(k)=base_rng(k)+1;
                            if rn(k)> cube_size(k)
                                continue;
                            end
                        case {4,5,6}
                            m = k-3;
                            rn = base_rng+1;
                            rn(m) = rn(m)-1;
                            if any(rn > cube_size)
                                continue;
                            end
                    end
                    tt = prod(rn);
                    if tt > ref_size
                        continue;
                    end
                    if tt > ttl
                        range = rn;
                        ttl = tt;
                    end
                end
                if ttl == base_ttl
                    break;
                end
                base_rng = range;
                base_ttl = ttl;
            end
            
            if iscolumn(range)
                range = range';
            end
            range = [range c];
        end
        
        function [prmt] = calcPermutSizes(prmt)
            if isfield(prmt, 'min_order')
                return;
            end
            
            switch prmt.PL_mode
                case SensingMatrixConvolve.MODE_ARBT
                    if ~isfield(prmt, 'N_msrs')
                        prmt.N_msrs = length(prmt.PL_seq);
                    end
                case SensingMatrixConvolve.MODE_1D  % NOP
                case SensingMatrixConvolve.MODE_LCLS % NOP
                case SensingMatrixConvolve.MODE_GLBL
                    prmt.PL_range = SensingMatrixConvolve.findPermutSizes(...
                        prmt.PL_range(1:3), prmt.PL_size, prmt.N_msrs);
                    prmt.N_msrs = prod(prmt.PL_range);
                        
            end
            prmt.min_order = prmt.N_msrs;
        end
    end
    
end

