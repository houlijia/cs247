classdef SensingMatrixSqr < SensingMatrix
  % SensingMatrixSqr An abstract class describing a sensing matrix derived
  % from a square matrix by selecting a subset of the rows and a permutation
  % of the columns.
  %
  % This class defines a m by n sensing matrix of the form:
  %     A=I_mp*Pleft*S*Pright*I_pn
  % Where:
  %   S is a square matrix of order p (p>=m, p>=n).
  %
  %   Pleft and Pright are permutation matrices of order p x p, which
  %   correspond to the permutations PL and PR respectively.  PL and PR
  %
  %   I_kl denotes the k by l matrix in which all entries are zero except
  %   the entries in the main diagonal which are one.  If k<=l then
  %   multiplying a vector of order l by I_kl is truncation to the
  %   first k entries.  If k>l then multiplying a vector of order
  %   l by I_kl is addking l-k zeroes at the end.
  %
  % A permutation of 1:p is a vector P.  We say that the permutation maps k
  % to P(k) and the inverse permutation, IP, maps P(k) to k.  The permutation
  % matrix associated, with P, Pmtrx, is a matrix such that if y=Pmtrx*x then
  % y(P(k))==x(k). It is easy to see that the entries in Pmtrx are zeroes
  % except for one value of 1 in each row and in each column and it is given
  % by Pmtrx(k,l)==(P(k)==l).  Note also that the inverse of Pmtrx
  % corresponds to the inverse permutation, IP, and equals the transpose of
  % Pmtrx.  In matlab, multiplication by a permutatation matrix are easy:
  % y=Pmtrx*x is given by y(P)=x.  Y=Pmtrx'*x is given by y(IP)=x or
  % equivalently, y=x(P).
  %
  % Let IPL and PR be the permutation corresponding to Pright and Pleft,
  % and let PL and IPR be their respective inverse permutations.
  % Consider y=A*x where x is a vector of order n.  y can be computed in
  %  several steps as follows:
  %      x0 = [x; zeroes(p-n,1)]
  %      u(PR) = x0
  %      v = S*u
  %      w(IPL) = v
  %      y = w(1:m)
  %  The last two steps may be replaced by one step using the inverse
  %  permutation:
  %      y = v(PL(1:m))
  %  A' = I_np*Pright'*S'*Pleft'*I_pm.  Therefore, similarly
  %  y=A'*x (now x is order m and y is of order n) can be computed
  %  similarly.  However, since the permutations need to be inversed we
  %  use them as indices on the left side of the assignement instead of
  %  in the right side:
  %      x0 = [x; zeroes(p-m,1)]
  %      u(PL) = x0 or equivalently u(IPL) = x0
  %      v = S'*u
  %      w(IPR) = v
  %      y = w(1:n)
  %  or equivalently
  %      y = v(PR(1:n))
  %
  
    properties (Constant)
    % Modes of selection of rows. In the following random may also mean
    % "arbitrary".
    SLCT_MODE_NONE = 0; % No randomization. Contiguous sequence of measurements 
                        % beginning from indx 1.
    SLCT_MODE_ARBT = 1; % Random indices
    SLCT_MODE_1D = 2; % A contiguous sequence of indices with a random starting point.
    SLCT_MODE_LCLS = 3; % Random indices where around each index there are indices
                        % which are physical neighbors (see
                        % SensingMatrixConvolve for more information.
    SLCT_MODE_GLBL = 4; % A group of indices with a random starting point designed
                        % to have maximum number of neighboring measurements(see
                        % SensingMatrixConvolve for more information.
    SLCT_MODE_ZGZG = 5; % Indices selected in a determinstic method using Xin Yuan's
                        % zigzag method (only for WH).
  end
  properties
    sqr_order=0;   % p in the explanation
    PL=[];      % Left permutation (only n_rows elements). Empty SLCT_MODE_NONE
    PR=[];      % right permutation. Empty if no right permutation.
    
    % A function handle to multiply a matrix with sqr_order rows by the square ...
    % matrix
    mltSqr = [];
    
    % A function handle to multiply a matrix with sqr_order rows by the...
    % transpose of the square matrix
    mltTrnspSqr = [];
  end
  
  methods (Abstract, Access=protected)
    % Default order of square matrix
    order = defaultOrder(obj, num_rows, num_columns, prmt_info)
  end
  
  methods
     function obj = SensingMatrixSqr(varargin)
       % Constructor. 
      % Input:
      %   opts: If not present, do nothing. Otherwise this is options struct. 
      %         Fields are:
      %           Fields for use by setSensingMatrix()
      %           sqr_order: (required) order of the square matrix
      %           PL: (required) the PL permutation
      %           PR: (required) the PR permutation
      %           mltSqr: (required) a function handle for obj.mltSqr
      %           mltTrnspSqr: (required) a function handle for
      %             obj.mltTrnspSqr'
      obj.setSensingMatrixSqr(varargin{:});
    end
    
    function set(obj, opts)
      % Input:
      %   obj: this object
      %   opts: If not present, do nothing. Otherwise this is options struct. 
      %         Fields are:
      %           Fields for use by setSensingMatrix()
      %           sqr_order: (required) order of the square matrix
      %           PL: (required) the PL permutation
      %           PR: (required) the PR permutation
      %           mltSqr: (required) a function handle for obj.mltSqr
      %           mltTrnspSqr: (required) a function handle for
      %             obj.mltTrnspSqr'
      if nargin > 1
        obj.setSensingMatrixSqr(opts);
      end
    end
    
    % Set order and permutations.
    function setPermutations(obj, order, PL, PR)
      obj.sqr_order = obj.toCPUIndex(order);
      if nargin > 2
        if ~isempty(PL)
          obj.PL = obj.toIndex(PL(1:obj.n_rows));
        else
          obj.PL = obj.toIndex([]);
        end
        obj.PR=obj.toIndex(PR(:));
      else
        obj.PL = obj.toIndex([]);
        obj.PR = obj.toIndex([]);
      end
    end
    
    function y=doMultVec(obj, x)
      % doMultVec - implemenentation of abstract method of
      % SensingMatrix - multiply a vector x by A.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      u = [x; zeros(obj.sqr_order-size(x,1), 1, 'like', x)];
      u(obj.PR) = u(1:length(obj.PR));
      v =obj.mltSqr(u);
      if ~isempty(obj.PL)
        y = v(obj.PL);
      else
        y = v(1:obj.n_rows);
      end
    end
    
    function y=doMultTrnspVec(obj,x)
      % doMultTrnspVec - implemenentation of abstract method of
      % SensingMatrix - multiply a vector x by A'.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      n = obj.n_cols;
      u = zeros(obj.sqr_order, 1, 'like', x);
      if ~isempty(obj.PL)
        u(obj.PL) = x;
      else
        u(1:obj.n_rows) = x;
      end
      v =obj.mltTrnspSqr(u);
      y = v(obj.PR(1:n));
    end
    
    function y=doMultMat(obj, x)
      u = [x; zeros(obj.sqr_order-size(x,1), size(x,2), 'like', x)];
      u(obj.PR,:) = u(1:length(obj.PR),:);
      v =obj.mltSqr(u);
      if ~isempty(obj.PL)
        y = v(obj.PL,:);
      else
        y = v(1:obj.n_rows,:);
      end
    end
    
    function y=doMultTrnspMat(obj,x)
      n = obj.n_cols;
      u = zeros(obj.sqr_order, size(x,2), 'like', x);
      if ~isempty(obj.PL)
        u(obj.PL,:) = x;
      else
        u(1:obj.n_rows,:) = x;
      end
      v =obj.mltTrnspSqr(u);
      y = v(obj.PR(1:n),:);      
    end
    
    function mtrx = getSqrMtrx(obj)
      mtrx = obj.mltSqr(eye(obj.sqr_order));
    end
    
    function msrs_noise= calcMsrsNoise(obj, n_orig_pxls, pxl_width)
      % Compute standard deviation of noise in measurement, assuming that
      % the noise results from pixel quantization
      %   Input:
      %     n_orig_pxls - number of original pixels
      %     pxl_width - width of pixel quantization interval
      %                 (optional, default = 1 for uniform)
      
      n_pxls = obj.nCols();
      mtrx_nrm = obj.norm();
      if nargin < 3
        pxl_width = 1;
        if nargin < 2
          n_orig_pxls = n_pxls;
        end
      end
      msrs_noise = mtrx_nrm * pxl_width * double(n_pxls)/...
        sqrt(double(12*n_orig_pxls)*double(obj.sqr_order));
    end
  end
  
  methods (Access=protected)
    function setSensingMatrixSqr(obj, opts)
      % Input:
      %   obj: this object
      %   opts: If not present, do nothing. Otherwise this is options struct. 
      %         Fields are:
      %           Fields for use by setSensingMatrix()
      %           sqr_order: (required) order of the square matrix
      %           PL: (required) the PL permutation
      %           PR: (required) the PR permutation
      %           mltSqr: (required) a function handle for obj.mltSqr
      %           mltTrnspSqr: (required) a function handle for
      %             obj.mltTrnspSqr'
      
      if nargin < 2
        return
      end
      obj.setSensingMatrix(opts);
      obj.mltSqr = opts.mltSqr;
      obj.mltTrnspSqr = opts.mltTrnspSqr;
      obj.setPermutations(opts.sqr_order, opts.PL, opts.PR);
    end
    
    function args = parseInitArgs(obj, args, names)
      if length(args)==1 && (isstruct(args{1}) || ischar(args{1}))
        if ischar(args{1})
          spec = ProcessingParams.parse_opts(args{1});
        else
          spec = args{1};
        end
        n_names = length(names);
        args = cell(1,n_names);
        for k=1:n_names
          if isfield(spec, names{k})
            args{k} = spec.(names{k});
          elseif strcmp(names{k},'order')
            if isfield(spec, 'num_rows')
              n_r = spec.num_rows;
            else
              n_r = [];
            end
            if isfield(spec, 'num_columns')
              n_c = spec.num_columns;
            else
              n_c = [];
            end
            if isfield(spec, 'prmt')
              p_i = spec.prmt;
            else
              p_i = [];
            end
            args{k} = obj.defaultOrder(n_r, n_c, p_i);
          else
            args = args(1:k-1);
            break;
          end
        end
      end
    end
    
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrix();
      obj.sqr_order = obj.toCPUIndex(obj.sqr_order);
      obj.PR = obj.toIndex(obj.PR);
    end
  end
  
 end
