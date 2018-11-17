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
    %     SLCT_MODE_NONE - No randomization. Contiguous sequence of measurements 
    %                      beginning from indx 1.
    %     SLCT_MODE_ARBT - arbitratry sequence provided by the field PL_seq.
    %                 In this case N_msrs is the length of that sequence.
    %                 Fields required: PL_seq.
    %     SLCT_MODE_1D   - A contiguous sequence starting at PL_start,
    %                 wrapping around sqr_order.
    %                 Fields required: PL_start, N_msrs.
    %     SLCT_MODE_LCLS - Make sure to have measurements with specified
    %                 offsets.  Let [v,h,t] be the first 3 fields in
    %                 PL_range and let
    %                     L(s)={s+i+V*(j+H*k) | 0<=i<v, 0<=j<h, 0<=k<t}
    %                 Consider the sequence obtained by concatenating
    %                 L(s(1)),L(s(2))),... where s(1),s(2) are given by
    %                   PL_seq.  Remove all repeated entries from this
    %                 sequence and truncate it after N_msrs entries.
    %                 This is the required sequence.
    %                 Fields required: PL_range, PL_seq, PL_size, N_msrs.
    %     SLCT_MODE_GLBL - Get a maximal number of measurements with
    %                 specified offsets. Let [v,h,t] be the first 3 fields
    %                 in PL_range.  These numbers are multiplied
    %                 proportinally to be as close as possible to
    %                 (V,H,T).  Then we create as many copies of that
    %                 sequence as needed to get as close to N_msrs, by
    %                 shifting by VHT. Finally, we add PL_start.
    %                 Fields required: PL_start, PL_range, PL_size, N_msrs.
    %   PL_seq - needed only in mode SLCT_MODE_ARBT. Is either a numerical
    %            sequence to be used as PL or a pointer to a function
    %            which, when called provides the sequence.
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
  
  properties (Access=private)
    seq_type=-1;  % Sequence type: -1=unknown 0=real, 1= integer,
    % 2= signed integer, 3=binary integer
    seq_vals=[];  % If the sequence is binary, these are the values it gets.
    
    seq_sum_sqr=[]; % rms of sequence values. empty indicates not computed yet.
    fft_order;
    fft_seq=[];
    fft_trnsp_seq=[];
  end
    
  properties (GetAccess=?SensingMatrix, SetAccess=protected)
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
    function obj = SensingMatrixConvolve(opts)
      % Constructor
      %   opts: If not present, do nothing. Otherwise this is options struct. 
      %         Required fields are:
      %           Fields for use by setSensingMatrixSqr()
      %           prmt - A struct of the same structure as obj.permut.
      %           sqnc - the sequence used to create the matrix. sqr_order is
      %                  the length of sqnc
      %      Note: At least one of opts.n_rows and prmt.N_msrs must be a defined
      %            non-negative numerical scalar. If both are defined and are 
      %            not empty they must be the same.
      if nargin > 0
        obj.setSensingMatrixConvolve(opts)
      end
    end
    
    function set(obj, opts)
      % Configure after creation
      % Input:
      %   obj: this object
      %   opts: If not present, do nothing. Otherwise this is options struct. 
      %         Required fields are:
      %           Fields for use by setSensingMatrixSqr()
      %           prmt - A struct of the same structure as obj.permut.
      %           sqnc - the sequence used to create the matrix. sqr_order is
      %                  the length of sqnc
      %      Note: At least one of opts.n_rows and prmt.N_msrs must be a defined
      %            non-negative numerical scalar. If both are defined and are 
      %            not empty they must be the same.
      if nargin > 0
        obj.setSensingMatrixConvolve(opts);
      end
    end
    
    function aux_info = setSequence(obj, sqnc, aux_info)
      if iscolumn(sqnc)
        obj.seq = obj.toFloat(sqnc);
      else
        obj.seq = obj.toFloat(sqnc');
      end
      
      if obj.use_fft
        if nargin >= 3
          obj.fft_seq = obj.toFloat(aux_info.fft_seq);
          obj.fft_trnsp_seq = obj.toFloat(aux_info.fft_trnsp_seq);
          obj.fft_order = obj.toIndex(aux_info.fft_order);
        else
          obj.calcFFTSeq();
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
        obj.seq_vals = obj.toFloat(sq_vals);
      end
    end
    
    function y=multSqr(obj,x)
      %Multiply the matrix x of size (sqr_order,:) by S
      if ~obj.use_fft
        y = zeros(size(x));
        for k=1:size(x,2)
          y(:,k) = conv(x(:,k), [obj.seq(2:end); obj.seq], 'same');
        end
      else
        y = obj.calcMultSqr(x,obj.fft_seq);
        if obj.use_fft == 2
          yc = zeros(size(x));
          for k=1:size(x,2)
            yc(:,k) = conv(x(:,k), [obj.seq(2:end); obj.seq], 'same');
          end
          if ~isequal(y,yc)
            err = norm(y-yc,inf);
            fprintf('SensingMatrixConvolve.multSqr diff: %g (%g)\n',...
              err, err/(norm(yc,inf)+1e-16));
          end
        end
      end
    end
    
    function y=multTrnspSqr(obj,x)
      %Multiply the matrix x of size (sqr_order,:) by S'
      if ~obj.use_fft
        y = zeros(size(x));
        for k=1:size(x,2)
          y(:,k) = conv(x(:,k), [obj.seq(end:-1:1);obj.seq(end:-1:2)], 'same');
        end
      else
        y = obj.calcMultSqr(x,obj.fft_trnsp_seq);
        if obj.use_fft == 2
          yc = zeros(size(x));
          for k=1:size(x,2)
            yc(:,k) = conv(x(:,k), [obj.seq(end:-1:1);obj.seq(end:-1:2)], 'same');
          end
          if ~isequal(y,yc)
            err = norm(y-yc,inf);
            fprintf('SensingMatrixConvolve.multTrnspSqr diff: %g (%g)\n',...
              err, err/(norm(yc,inf)+1e-16+norm(yc,inf)));
          end
        end
      end
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
    
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(obj,y)
      if obj.is_transposed
        y = obj.normalizeMsrs@SensingMatrix(y);
      else
        if isempty(obj.seq_sum_sqr)
          obj.seq_sum_sqr = double(dot(obj.seq, obj.seq));
        end
        y = y / sqrt(obj.seq_rms);
      end
    end
    
    % undo the operation of normalizeMsrs
    function y = deNormalizeMsrs(obj,y)
      if obj.is_transposed
        y = obj.deNormalizeMsrs@SensingMatrix(y);
      else
        if isempty(obj.seq_sum_sqr)
          obj.seq_sum_sqr = double(dot(obj.seq, obj.seq));
        end
        y = y * sqrt(obj.seq_rms);
      end
    end
    
  end   % methods
  
  methods (Access=protected)
    function setSensingMatrixConvolve(obj, opts)
      % Input:
      %   obj: this object
      %   opts: If not present, do nothing. Otherwise this is options struct. 
      %         Required fields are:
      %           Fields for use by setSensingMatrixSqr()
      %           prmt - A struct of the same structure as obj.permut.
      %           sqnc - the sequence used to create the matrix. sqr_order is
      %                  the length of sqnc
      %      Note: At least one of opts.n_rows and prmt.N_msrs must be a defined
      %            non-negative numerical scalar. If both are defined and are 
      %            not empty they must be the same.
      
      if nargin < 2
        return
      end
      
      opts.prmt = SensingMatrixConvolve.calcPermutSizes(opts.prmt);
      
      if isfield(opts.prmt, 'N_msrs') && ~isempty(opts.prmt.N_msrs)
        if isfield(opts, 'n_rows') && ~isempty(opts.n_rows)
          if ~isequal(opts.prmt.N_msrs, opts.n_rows)
            error('opts.n_rows and opts.prmt.N_msrs are different');
          end
        else
          opts.n_rows = opts.prmt.N_msrs;
        end
      elseif isfield(opts, 'n_rows') && ~isempty(opts.n_rows)
        opts.prmt.N_msrs = opts.n_rows;
      end
      
      opts.sqr_order = opts.n_cols;
      opts.PL = SensingMatrixConvolve.calcPL(opts.prmt, length(opts.sqnc));
      opts.PR = SensingMatrixConvolve.calcPR(opts.prmt, length(opts.sqnc));
      opts.mltSqr = @obj.multSqr;
      opts.mltTrnspSqr = @obj.multTrnspSqr;
      
      obj.setSensingMatrixSqr(opts);
      obj.setSequence(opts.sqnc);
      obj.permut = opts.prmt;
      [sq_type, sq_vals] = obj.calcSeqType(opts.sqnc);
      obj.setSeqType(sq_type, sq_vals);
     
    end
    
    function ord = defaultOrder(~, ~, num_columns, prmt_info)
      if nargin < 4
        prmt_info = obj.permut;
      end
      ord = SensingMatrixConvolve.calcDefaultOrder(num_columns,prmt_info);
    end
    
    function [PL, PR] = makePermutations(obj, order, opts)
      prmt = opts.prmt;
      PL = SensingMatrixConvolve.calcPL(prmt, order);
      PR = SensingMatrixConvolve.calcPR(prmt, order, obj.n_cols);
      obj.permut = prmt;
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
    %           obj.PL(inp_list(j))-obj.PL(inp_list(i)) =
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
        ipl = obj.PL;
      end
      
      % Start assuming that all indices will fit
      indcs = obj.toIndex(zeros(length(inp_list), lofst));
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
    
    function compSeqRms(obj)
      obj.seq_rms = norm(obj.seq,2);
    end
    
    function calcFFTSeq(obj)
      N=length(obj.seq);
      obj.fft_order = pow2(nextpow2(N*2));
      fft_len = obj.fft_order/2 + 1;
      SQ = fft([obj.seq(2:end); obj.seq;...
        obj.zeros(obj.fft_order-(2*N-1),1)]);
      obj.fft_seq = SQ(1:fft_len);
      SQ = fft([obj.seq(end:-1:1);obj.seq(end:-1:2);...
        obj.zeros(obj.fft_order-(2*N-1),1)]);
      obj.fft_trnsp_seq = SQ(1:fft_len);
    end
    
    % This is an upper bound approximation.
    function y = compNorm(obj)
      if isempty(obj.seq)
        error('obj.seq not specified');
      end
      if obj.use_fft
        if isempty(obj.fft_seq)
          obj.calcFFTSeq();
        end
        fseq = obj.fft_seq;
      else
        N=length(obj.seq);
        ordr = pow2(nextpow2(N*2));
        sq = [obj.seq(end:-1:1);obj.seq(end:-1:2);obj.zeros(ordr-(2*N-1),1)];
        fseq = fft(sq);
      end
      y = norm(fseq,inf);
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqr();
      obj.seq_type = obj.toSInt(obj.seq_type);
      obj.fft_order = obj.toIndex(obj.fft_order);
      obj.use_fft = obj.toIndex(obj.use_fft);
    end
    
    function setCastFloat(obj)
      obj.setCastFloat@SensingMatrixSqr();
      obj.seq = obj.toFloat(obj.seq);
      obj.seq_vals = obj.toFloat(obj.seq_vals);
      obj.fft_seq = obj.toFloat(obj.fft_seq);
      obj.fft_trnsp_seq = obj.toFloat(obj.fft_trnsp_seq);
    end
  end
  
  methods (Access=private)
    %Multiply the vector x of size (sqr_order,1) by S
    function y=calcMultSqr(obj,x,SQ)
      if length(x) < obj.fft_order
        x = [x;obj.zeros(obj.fft_order-length(x),size(x,2))];
      end
      X = fft(x);
      Y = (SQ * obj.ones(1,size(X,2))) .* X(1:length(SQ),:);
      y = real(ifft([Y; conj(Y(end-1:-1:2,:))]));
      y = y(obj.sqr_order:2*obj.sqr_order-1,:);
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
      sqln = double(sqln);
      switch prmt.PL_mode
        case SensingMatrixConvolve.SLCT_MODE_NONE
          plseq = [];
        case SensingMatrixConvolve.SLCT_MODE_ARBT
          plseq = double(prmt.PL_seq);
          if ~iscolumn(plseq)
            plseq = plseq';
          end
          prmt.N_msrs = length(plseq);
        case SensingMatrixConvolve.SLCT_MODE_1D
          plseq = prmt.PL_start + (1:prmt.N_msrs)';
          plseq = 1 + mod(plseq(:)-1,sqln);
        case SensingMatrixConvolve.SLCT_MODE_LCLS
          plseq = SensingMatrixConvolve.calcSlctPL(...
            prmt.PL_range(1:3), prmt.PL_size);
          
          plseq = plseq * ones(1,length(prmt.PL_seq)) + ...
            ones(size(plseq))*prmt.PL_seq;
          plseq = 1 + mod(plseq(:)-1,sqln);
          plseq = unique(plseq, 'stable');
          plseq = plseq(1:prmt.N_msrs);
        case SensingMatrixConvolve.SLCT_MODE_GLBL
          prmt.PL_range = SensingMatrixConvolve.findPermutSizes(...
            prmt.PL_range(1:3), prmt.PL_size, prmt.N_msrs);
          plseq = SensingMatrixConvolve.calcSlctPL(...
            prmt.PL_range, prmt.PL_size);
          plseq = plseq + prmt.PL_start;
          plseq = 1 + mod(plseq(:)-1,double(sqln));
      end
      
      pl_prmt =  plseq(:);
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
      n_msr = double(n_msrs);
      c = ceil(n_msr/prod(cube_size));
      ref_size = n_msr/c;
      f = (n_msr/(c*prod(base_rng)))^(1/3);
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
        case SensingMatrixConvolve.SLCT_MODE_NONE % NOP
        case SensingMatrixConvolve.SLCT_MODE_ARBT
          if ~isfield(prmt, 'N_msrs')
            prmt.N_msrs = length(prmt.PL_seq);
          end
        case SensingMatrixConvolve.SLCT_MODE_1D  % NOP
        case SensingMatrixConvolve.SLCT_MODE_LCLS % NOP
        case SensingMatrixConvolve.SLCT_MODE_GLBL
          prmt.PL_range = SensingMatrixConvolve.findPermutSizes(...
            prmt.PL_range(1:3), prmt.PL_size, prmt.N_msrs);
          prmt.N_msrs = prod(prmt.PL_range);
          
      end
      prmt.min_order = prmt.N_msrs;
    end
  end
  
end

