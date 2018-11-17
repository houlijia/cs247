classdef SensingMatrixMLSeq < SensingMatrixCnvlvRnd
  %SensingMatrixLFSR convolutional matrix based on maximum length sequence.
  
  properties (Constant)
    % Determine whehter to use Fast-Walsh Hadamard transform for the
    % convolution:
    %   0 - do not use WH transform. Use methods of superclass
    %   1 - Use WH transform.
    %   2 - Use WH transform and check it against corresponding methods
    %       of superclass.
    use_fwht = 1;
    
    % If WH transform is used, determines whether to use
    % Matlab' built in fast WH functions or the .mex files.
    use_matlab_WHtransform = false;
  end
  
  properties
    log2order;
  end
    
  properties (Access=private)
    fwh_left = [];
    fwh_right = [];
    fwh_fun
  end
  
  methods
    % Constructor
    %   Input:
    %     prmt - A permutations specification struct, same as
    %            SensingMatrixConvolve.permut.
    %     num_columns - number of columns
    %     rnd_seed - Random number generator seed
    %     order - order of the sequence (= dimension of square matrix)
    %   rnd_type - type of random number generator. Can be
    %        string - RandStream type
    %        0 - default type
    %        1 - Use RndCStrm
    %        2 - Use RndCStrm and compare with RandStream
    function obj = SensingMatrixMLSeq(varargin)
      obj.setSensingMatrixMLSeq(varargin{:});
    end
    
    % Initialize matrix
    %   Input:
    %     obj - this object
    %     prmt - A permutations specification struct, same as
    %            SensingMatrixConvlolve.permut.
    %     num_columns - number of columns
    %     rnd_seed - Random number generator seed
    %     order - order of the sequence (= dimension of square matrix)
    %   rnd_type - type of random number generator. Can be
    %        string - RandStream type
    %        0 - default type
    %        1 - Use RndCStrm
    %        2 - Use RndCStrm and compare with RandStream
    function set(obj, varargin)
      varargin = parseInitArgs(varargin, {'prmt', 'num_columns', ...
        'rnd_seed', 'order', 'rnd_type'});
      obj.setSensingMatrixMLSeq(varargin{:});
    end
    
    function makeSequence(obj)
      persistent seqs;
      
      if isempty(seqs)
        seqs = cell(1,32);
      end
      
      obj.log2order = obj.toCPUIndex(nextpow2(obj.sqr_order));
      
      if isempty(seqs{obj.log2order})
        fname = sprintf('lfsr_%d_seq.dat', obj.log2order);
        [fh, emsg] = fopen(fname, 'r');
        if fh == -1
          error('Failed opening %s (%s)\n', fname, emsg);
        end
        
        cnt = ceil(obj.toCPUFloat(obj.sqr_order/8));
        [bytes, len] = fread(fh, cnt, '*uint8');
        if len<cnt
          if feof(fh)
            error('Failed reading %s (reached EOF)\n', fname);
          else
            error('Failed reading %s (%s)\n', fname, ferror(fh));
          end
        end
        
        sq = obj.bit_tbl(uint32(bytes)+1, :);
        sq = sq';
        sq = sq(1:obj.sqr_order);
        sq = 1-2*sq;
        seq_data = struct('seq',sq);
        seq_data.aux = obj.setSequence(sq);
        
        if obj.use_fwht
          fname = sprintf('WH_lfsr_%d_indcs.dat', obj.log2order);
          [fh, emsg] = fopen(fname, 'r');
          if fh == -1
            error('Failed opening %s (%s)\n', fname, emsg);
          end
          
          cnt = obj.sqr_order;
          [fwh_data, len] = fread(fh, [2,cnt], '*uint32', 0, 'ieee-be');
          if len<cnt
            if feof(fh)
              error('Failed reading %s (reached EOF)\n', fname);
            else
              error('Failed reading %s (%s)\n', fname, ferror(fh));
            end
          end
          seq_data.fwh_left = 1+fwh_data(1,:)';
          seq_data.fwh_right = 1+fwh_data(2,:)';
          
          if obj.use_matlab_WHtransform
            seq_data.fwh_fun = ...
              @(x) ifwht(x, obj.sqr_order+1, 'hadamard');
          else
            seq_data.fwh_fun = @wht_mex;
          end
        end
        
        seqs{obj.log2order} = seq_data;
      else
        seq_data = seqs{obj.log2order};
        obj.setSequence(seq_data.seq, seq_data.aux);
      end
      
      if obj.use_fwht
        obj.fwh_left = obj.toIndex(seq_data.fwh_left);
        obj.fwh_right = obj.toIndex(seq_data.fwh_right);
        obj.fwh_fun = seq_data.fwh_fun;
      end
      
      obj.setSeqType(3, [1 -1]);
    end
    
    function y=multSqr(obj,x)
      %Multiply the vector x of size (sqr_order,1) by S
      if ~obj.use_fwht
        y = obj.multSqr@SensingMatrixCnvlvRnd(x);
      else
        xm(obj.fwh_right,:) = x;
        ym = obj.fwh_fun(xm(:));
        y = ym(obj.fwh_left);
        if obj.use_fwht == 2
          yc = obj.multSqr@SensingMatrixCnvlvRnd(x);
          if ~isequal(y,yc)
            err = norm(y-yc,inf);
            fprintf('SensingMatrixMLSeq.multSqr diff: %g (%g)\n',...
              err, err/(norm(yc,inf)+1e-16));
          end
        end
      end
    end
    
    function y=multSqrMat(obj,x)
      %Multiply the matrix x of size (sqr_order,1) by S
      if ~obj.use_fwht
        rw = (1:size(x,2));
        yc = arrayfun(@(k) {obj.multSqr(x(:,k))}, rw);
        y = horzcat(yc{:});
      else
        xm(obj.fwh_right,:) = x;
        ym = obj.fwh_fun(xm);
        y = ym(obj.fwh_left,:);
        if obj.use_fwht == 2
          yc = obj.multSqrMat@SensingMatrixCnvlvRnd(x);
          if ~isequal(y,yc)
            err = norm(y(:)-yc(:),inf);
            fprintf('SensingMatrixMLSeq.multSqrMat diff: %g (%g)\n',...
              err, err/(norm(yc,inf)+1e-16));
          end
        end
      end
    end
    
    function y=multTrnspSqr(obj,x)
      %Multiply the vector x of size (sqr_order,1) by S'
      if ~obj.use_fwht
        y= obj.multTrnspSqr@SensingMatrixCnvlvRnd(x);
      else
        xm = obj.zeros(length(x)+1,1);
        xm(obj.fwh_left,1) = x;
        ym = obj.fwh_fun(xm(:));
        y = ym(obj.fwh_right,:);
        if obj.use_fwht == 2
          yc = multTrnspSqr@SensingMatrixCnvlvRnd(obj,x);
          if ~isequal(y,yc)
            err = norm(y-yc,inf);
            fprintf('SensingMatrixMLSeq.multTrnspSqr diff: %g (%g)\n',...
              err, err/(norm(yc,inf)+1e-16));
          end
        end
      end
    end
    
    function y=multTrnspSqrMat(obj,x)
      %Multiply the matrix x of size (sqr_order,1) by S'
      if ~obj.use_fwht
        rw = (1:size(x,2));
        yc = arrayfun(@(k) {obj.multTrnspSqr(x(:,k))}, rw);
        y = horzcat(yc{:});
      else
         xm = obj.zeros(length(x)+1,size(x,2));
         xm(obj.fwh_left,:) = x;
         ym = obj.fwh_fun(xm);
         y = ym(obj.fwh_right,:);
         if obj.use_fwht == 2
           yc = obj.multTrnspSqrMat@SensingMatrixCnvlvRnd(x);
           if ~isequal(y,yc)
             err = norm(y(:)-yc(:),inf);
             fprintf('SensingMatrixMLSeq.multTrnspSqrMat diff: %g (%g)\n',...
               err, err/(norm(yc,inf)+1e-16));
          end
        end
      end
    end
    
    function y = cmpExactNorm(obj)
      if obj.sqr_order > 1
        y = sqrt(obj.toCPUFloat(obj.sqr_order+1));
      else
        y = 1;
      end
    end

    function chkOrtho(obj)
      obj.setPsdOrtho_row(false);
      obj.setPsdOrtho_col(false);
    end
    

  end
  
  methods (Access=protected)
    % Initialize matrix
    %   Input:
    %     obj - this object
    %     prmt - A permutations specification struct, same as
    %            SensingMatrixConvlolve.permut.
    %     num_columns - number of columns
    %     rnd_seed - Random number generator seed
    %     order - order of the sequence (= dimension of square matrix)
    %   rnd_type - type of random number generator. Can be
    %        string - RandStream type
    %        0 - default type
    %        1 - Use RndCStrm
    %        2 - Use RndCStrm and compare with RandStream
    function setSensingMatrixMLSeq(obj, prmt, num_columns,...
        rnd_seed, order, rnd_type)
      if obj.use_fwht == 1
        obj.use_fft = 0;
      end
      
      if nargin < 3
        smr_args = {};
      elseif nargin == 4
        order = obj.defaultOrder(...
          prmt.N_msrs, num_columns, prmt);
        if order < num_columns
          error('order=%d < than num_columns=%5d', order, num_columns);
        end
      end
      
      switch nargin
        case 3
          smr_args = { prmt, num_columns };
        case {4,5}
          smr_args = { prmt, num_columns,  rnd_seed, order};
        case 6
          smr_args = { prmt, num_columns, ...
            rnd_seed, order, rnd_type};
      end
      obj.setSensingMatrixCnvlvRnd(smr_args{:});
    end
    
    function order = defaultOrder(obj, ~, num_columns, prmt)
      order = SensingMatrixMLSeq.calcDefaultOrder(...
        num_columns, prmt);
      order = obj.toCPUIndex(order);
    end
        
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixCnvlvRnd();
      obj.log2order = obj.toCPUIndex(obj.log2order);
      obj.fwh_left = obj.toIndex(obj.fwh_left);
      obj.fwh_right = obj.toIndex(obj.fwh_right);
    end
    
    function y = compNorm(obj)
      y = obj.cmpExactNorm();
    end
  end
  
  methods (Static)
    function order = calcDefaultOrder(num_columns, prmt)
      if ~isfield(prmt, 'min_order')
        prmt = SensingMatrixCnvlvRnd.calcPermutSizes(prmt);
      end
      order = SensingMatrixConvolve.calcDefaultOrder(num_columns, prmt);
      order = pow2(double(nextpow2(order+1))) - 1;
    end
  end
  
  properties (Constant, Access=private)
    bit_tbl = [...
      0 0 0 0 0 0 0 0;
      1 0 0 0 0 0 0 0;
      0 1 0 0 0 0 0 0;
      1 1 0 0 0 0 0 0;
      0 0 1 0 0 0 0 0;
      1 0 1 0 0 0 0 0;
      0 1 1 0 0 0 0 0;
      1 1 1 0 0 0 0 0;
      0 0 0 1 0 0 0 0;
      1 0 0 1 0 0 0 0;
      0 1 0 1 0 0 0 0;
      1 1 0 1 0 0 0 0;
      0 0 1 1 0 0 0 0;
      1 0 1 1 0 0 0 0;
      0 1 1 1 0 0 0 0;
      1 1 1 1 0 0 0 0;
      0 0 0 0 1 0 0 0;
      1 0 0 0 1 0 0 0;
      0 1 0 0 1 0 0 0;
      1 1 0 0 1 0 0 0;
      0 0 1 0 1 0 0 0;
      1 0 1 0 1 0 0 0;
      0 1 1 0 1 0 0 0;
      1 1 1 0 1 0 0 0;
      0 0 0 1 1 0 0 0;
      1 0 0 1 1 0 0 0;
      0 1 0 1 1 0 0 0;
      1 1 0 1 1 0 0 0;
      0 0 1 1 1 0 0 0;
      1 0 1 1 1 0 0 0;
      0 1 1 1 1 0 0 0;
      1 1 1 1 1 0 0 0;
      0 0 0 0 0 1 0 0;
      1 0 0 0 0 1 0 0;
      0 1 0 0 0 1 0 0;
      1 1 0 0 0 1 0 0;
      0 0 1 0 0 1 0 0;
      1 0 1 0 0 1 0 0;
      0 1 1 0 0 1 0 0;
      1 1 1 0 0 1 0 0;
      0 0 0 1 0 1 0 0;
      1 0 0 1 0 1 0 0;
      0 1 0 1 0 1 0 0;
      1 1 0 1 0 1 0 0;
      0 0 1 1 0 1 0 0;
      1 0 1 1 0 1 0 0;
      0 1 1 1 0 1 0 0;
      1 1 1 1 0 1 0 0;
      0 0 0 0 1 1 0 0;
      1 0 0 0 1 1 0 0;
      0 1 0 0 1 1 0 0;
      1 1 0 0 1 1 0 0;
      0 0 1 0 1 1 0 0;
      1 0 1 0 1 1 0 0;
      0 1 1 0 1 1 0 0;
      1 1 1 0 1 1 0 0;
      0 0 0 1 1 1 0 0;
      1 0 0 1 1 1 0 0;
      0 1 0 1 1 1 0 0;
      1 1 0 1 1 1 0 0;
      0 0 1 1 1 1 0 0;
      1 0 1 1 1 1 0 0;
      0 1 1 1 1 1 0 0;
      1 1 1 1 1 1 0 0;
      0 0 0 0 0 0 1 0;
      1 0 0 0 0 0 1 0;
      0 1 0 0 0 0 1 0;
      1 1 0 0 0 0 1 0;
      0 0 1 0 0 0 1 0;
      1 0 1 0 0 0 1 0;
      0 1 1 0 0 0 1 0;
      1 1 1 0 0 0 1 0;
      0 0 0 1 0 0 1 0;
      1 0 0 1 0 0 1 0;
      0 1 0 1 0 0 1 0;
      1 1 0 1 0 0 1 0;
      0 0 1 1 0 0 1 0;
      1 0 1 1 0 0 1 0;
      0 1 1 1 0 0 1 0;
      1 1 1 1 0 0 1 0;
      0 0 0 0 1 0 1 0;
      1 0 0 0 1 0 1 0;
      0 1 0 0 1 0 1 0;
      1 1 0 0 1 0 1 0;
      0 0 1 0 1 0 1 0;
      1 0 1 0 1 0 1 0;
      0 1 1 0 1 0 1 0;
      1 1 1 0 1 0 1 0;
      0 0 0 1 1 0 1 0;
      1 0 0 1 1 0 1 0;
      0 1 0 1 1 0 1 0;
      1 1 0 1 1 0 1 0;
      0 0 1 1 1 0 1 0;
      1 0 1 1 1 0 1 0;
      0 1 1 1 1 0 1 0;
      1 1 1 1 1 0 1 0;
      0 0 0 0 0 1 1 0;
      1 0 0 0 0 1 1 0;
      0 1 0 0 0 1 1 0;
      1 1 0 0 0 1 1 0;
      0 0 1 0 0 1 1 0;
      1 0 1 0 0 1 1 0;
      0 1 1 0 0 1 1 0;
      1 1 1 0 0 1 1 0;
      0 0 0 1 0 1 1 0;
      1 0 0 1 0 1 1 0;
      0 1 0 1 0 1 1 0;
      1 1 0 1 0 1 1 0;
      0 0 1 1 0 1 1 0;
      1 0 1 1 0 1 1 0;
      0 1 1 1 0 1 1 0;
      1 1 1 1 0 1 1 0;
      0 0 0 0 1 1 1 0;
      1 0 0 0 1 1 1 0;
      0 1 0 0 1 1 1 0;
      1 1 0 0 1 1 1 0;
      0 0 1 0 1 1 1 0;
      1 0 1 0 1 1 1 0;
      0 1 1 0 1 1 1 0;
      1 1 1 0 1 1 1 0;
      0 0 0 1 1 1 1 0;
      1 0 0 1 1 1 1 0;
      0 1 0 1 1 1 1 0;
      1 1 0 1 1 1 1 0;
      0 0 1 1 1 1 1 0;
      1 0 1 1 1 1 1 0;
      0 1 1 1 1 1 1 0;
      1 1 1 1 1 1 1 0;
      0 0 0 0 0 0 0 1;
      1 0 0 0 0 0 0 1;
      0 1 0 0 0 0 0 1;
      1 1 0 0 0 0 0 1;
      0 0 1 0 0 0 0 1;
      1 0 1 0 0 0 0 1;
      0 1 1 0 0 0 0 1;
      1 1 1 0 0 0 0 1;
      0 0 0 1 0 0 0 1;
      1 0 0 1 0 0 0 1;
      0 1 0 1 0 0 0 1;
      1 1 0 1 0 0 0 1;
      0 0 1 1 0 0 0 1;
      1 0 1 1 0 0 0 1;
      0 1 1 1 0 0 0 1;
      1 1 1 1 0 0 0 1;
      0 0 0 0 1 0 0 1;
      1 0 0 0 1 0 0 1;
      0 1 0 0 1 0 0 1;
      1 1 0 0 1 0 0 1;
      0 0 1 0 1 0 0 1;
      1 0 1 0 1 0 0 1;
      0 1 1 0 1 0 0 1;
      1 1 1 0 1 0 0 1;
      0 0 0 1 1 0 0 1;
      1 0 0 1 1 0 0 1;
      0 1 0 1 1 0 0 1;
      1 1 0 1 1 0 0 1;
      0 0 1 1 1 0 0 1;
      1 0 1 1 1 0 0 1;
      0 1 1 1 1 0 0 1;
      1 1 1 1 1 0 0 1;
      0 0 0 0 0 1 0 1;
      1 0 0 0 0 1 0 1;
      0 1 0 0 0 1 0 1;
      1 1 0 0 0 1 0 1;
      0 0 1 0 0 1 0 1;
      1 0 1 0 0 1 0 1;
      0 1 1 0 0 1 0 1;
      1 1 1 0 0 1 0 1;
      0 0 0 1 0 1 0 1;
      1 0 0 1 0 1 0 1;
      0 1 0 1 0 1 0 1;
      1 1 0 1 0 1 0 1;
      0 0 1 1 0 1 0 1;
      1 0 1 1 0 1 0 1;
      0 1 1 1 0 1 0 1;
      1 1 1 1 0 1 0 1;
      0 0 0 0 1 1 0 1;
      1 0 0 0 1 1 0 1;
      0 1 0 0 1 1 0 1;
      1 1 0 0 1 1 0 1;
      0 0 1 0 1 1 0 1;
      1 0 1 0 1 1 0 1;
      0 1 1 0 1 1 0 1;
      1 1 1 0 1 1 0 1;
      0 0 0 1 1 1 0 1;
      1 0 0 1 1 1 0 1;
      0 1 0 1 1 1 0 1;
      1 1 0 1 1 1 0 1;
      0 0 1 1 1 1 0 1;
      1 0 1 1 1 1 0 1;
      0 1 1 1 1 1 0 1;
      1 1 1 1 1 1 0 1;
      0 0 0 0 0 0 1 1;
      1 0 0 0 0 0 1 1;
      0 1 0 0 0 0 1 1;
      1 1 0 0 0 0 1 1;
      0 0 1 0 0 0 1 1;
      1 0 1 0 0 0 1 1;
      0 1 1 0 0 0 1 1;
      1 1 1 0 0 0 1 1;
      0 0 0 1 0 0 1 1;
      1 0 0 1 0 0 1 1;
      0 1 0 1 0 0 1 1;
      1 1 0 1 0 0 1 1;
      0 0 1 1 0 0 1 1;
      1 0 1 1 0 0 1 1;
      0 1 1 1 0 0 1 1;
      1 1 1 1 0 0 1 1;
      0 0 0 0 1 0 1 1;
      1 0 0 0 1 0 1 1;
      0 1 0 0 1 0 1 1;
      1 1 0 0 1 0 1 1;
      0 0 1 0 1 0 1 1;
      1 0 1 0 1 0 1 1;
      0 1 1 0 1 0 1 1;
      1 1 1 0 1 0 1 1;
      0 0 0 1 1 0 1 1;
      1 0 0 1 1 0 1 1;
      0 1 0 1 1 0 1 1;
      1 1 0 1 1 0 1 1;
      0 0 1 1 1 0 1 1;
      1 0 1 1 1 0 1 1;
      0 1 1 1 1 0 1 1;
      1 1 1 1 1 0 1 1;
      0 0 0 0 0 1 1 1;
      1 0 0 0 0 1 1 1;
      0 1 0 0 0 1 1 1;
      1 1 0 0 0 1 1 1;
      0 0 1 0 0 1 1 1;
      1 0 1 0 0 1 1 1;
      0 1 1 0 0 1 1 1;
      1 1 1 0 0 1 1 1;
      0 0 0 1 0 1 1 1;
      1 0 0 1 0 1 1 1;
      0 1 0 1 0 1 1 1;
      1 1 0 1 0 1 1 1;
      0 0 1 1 0 1 1 1;
      1 0 1 1 0 1 1 1;
      0 1 1 1 0 1 1 1;
      1 1 1 1 0 1 1 1;
      0 0 0 0 1 1 1 1;
      1 0 0 0 1 1 1 1;
      0 1 0 0 1 1 1 1;
      1 1 0 0 1 1 1 1;
      0 0 1 0 1 1 1 1;
      1 0 1 0 1 1 1 1;
      0 1 1 0 1 1 1 1;
      1 1 1 0 1 1 1 1;
      0 0 0 1 1 1 1 1;
      1 0 0 1 1 1 1 1;
      0 1 0 1 1 1 1 1;
      1 1 0 1 1 1 1 1;
      0 0 1 1 1 1 1 1;
      1 0 1 1 1 1 1 1;
      0 1 1 1 1 1 1 1;
      1 1 1 1 1 1 1 1
      ];
  end
end

