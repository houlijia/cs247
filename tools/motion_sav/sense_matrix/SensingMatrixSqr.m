classdef SensingMatrixSqr < SensingMatrix
    % SensingMatrixSqr An abstract class describing a sensing matrix derived
    % from a square matrix by selecting a subset of the rows and a subset
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
    % Let PL and PR be the permutation corresponding to Pright and Pleft, 
    % and let IPL and IPR be their respective inverse permutations.
    % Consider y=A*x where x is a vector of order n.  y can be computed in
    %  several steps as follows:
    %      x0 = [x; zeroes(p-n,1)]
    %      u(PR) = x0
    %      v = S*u
    %      w(PL) = v
    %      y = w(1:m)
    %  The last two steps may be replaced by one step using the inverse
    %  permutation:
    %      y = v(IPL(1:m))
    %  A' = I_np*Pright'*S'*Pleft'*I_pm.  Therefore, similarly
    %  y=A'*x (now x is order m and y is of order n) can be computed
    %  similarly.  However, since the permutations need to be inversed we
    %  use them as indices on the left side of the assignement instead of
    %  in the right side:
    %      x0 = [x; zeroes(p-m,1)]
    %      u(IPL) = x0 or equivalently u = x0(PL)
    %      v = S'*u
    %      w(IPR) = v
    %      y = w(1:n)
    %  or equivalently
    %      y = v(PR(1:n))
    % 
    
    properties
        sqr_order=0;   % p in the explanation
        IPL=[];      % Inverse of left permutation
        PR=[];      % right permutation
    end
    
    methods (Abstract)
        %Multiply the vector x of size (sqr_order,1) by S
        y=multSqr(obj,x)

        %Multiply the vector x of size (sqr_order,1) by S'
        y=multTrnspSqr(obj,x)
        
        % Default order of square matrix
        order = defaultOrder(obj, num_rows, num_columns, prmt_info)
    end
    
    methods
        % Constructor. Can have either 2 or 5 arguments or none
        function obj = SensingMatrixSqr(varargin)
            obj.setSensingMatrixSqr(varargin{:});
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
                'order', 'PL', 'PR'});
            obj.setSensingMatrixSqr(varargin{:});
        end
        
        % Set order and permutations.
        function setPermutations(obj, order, PL, PR)
            obj.sqr_order = order;
            if nargin > 2
                if ~iscolumn(PL)
                    PL=PL';
                end
                obj.IPL(PL,1) = double((1:order)');
                if ~iscolumn(PR)
                    PR = PR';
                end
                obj.PR=double(PR);
            else
                obj.IPL = [];
                obj.PR = [];
            end
        end
           
        % doMultVec - implemenentation of abstract method of SensingMatrix -
        % multiply a vector x by A.
        % INPUT
        %    obj - this object
        %    x   - input vector.  The vector length need not be the matrix's
        %          number of columns, as long as it does not exceed 
        %          obj.sqr_order.
        %    m     (optional) dimension of output (must be <= obj.sqr_order).
        %          If not specified it is the number of rows.
        function y=doMultVec(obj, x, m)
            if nargin < 3
                m = obj.n_rows;
            end
            u = [x; zeros(obj.sqr_order-size(x,1), size(x,2))];
            u(obj.PR,:) = u(1:length(obj.PR),:);
            v =obj.multSqr(u);
            y = v(obj.IPL(1:m),:);
        end
            
        % doMultTrnspVec - implemenentation of abstract method of SensingMatrix -
        % multiply a vector x by A'.
        % INPUT
        %    obj - this object
        %    x   - input vector.  The vector length need not be the number
        %          of rows in A, as long as it does not exceed 
        %          obj.sqr_order.
        %    n     (optional) dimension of output (must be <= obj.sqr_order).
        %          If not specified it is the number of columns in A.
        function y=doMultTrnspVec(obj,x,n)
            if nargin < 3
                n = obj.n_cols;
            end
            u = zeros(obj.sqr_order, size(x,2));
            u(obj.IPL,1) = [x; zeros(obj.sqr_order-length(x),size(x,2))];
            v =obj.multTrnspSqr(u);
            y = v([obj.PR(1:min(n,length(obj.PR))); (length(obj.PR)+1):n],:);
        end
        
        function y = multSqrMat(obj,x)
            y = zeros(obj.sqr_order, size(x,2));
            for k=1:size(x,2)
                y(:,k) = obj.multSqr(x(:,k));
            end
        end
        
        function mtrx = getSqrMtrx(obj)
            mtrx = obj.multSqrMat(eye(obj.sqr_order));
        end
        
        function y = multSqrTrnspMat(obj,x)
            y = zeros(obj.sqr_order, size(x,2));
            for k=1:size(x,2)
                y(:,k) = obj.multTrnspSqr(x(:,k));
            end
        end
                
        function len=encode(obj, code_dst, info)
            % encode Basic SensingMatrix info
            len = obj.encode@SensingMatrix(code_dst, info);
            if ischar(len)
                return;
            end
            cnt = len;

            % Encode order of square matrix, its type and permut
            len = code_dst.writeUInt(obj.sqr_order);
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
            n_read = obj.decode@SensingMatrix(code_src, info, cnt);
            len = n_read;
            if ischar(n_read)
                return;
            end
            cnt = cnt - n_read;
            
            %Decode  order and type
            [order, n_read] = code_src.readUInt(cnt);
            if ischar(order)
                len = order;
                return;
            end
            len = len + n_read;
        end
    end        
    
    methods (Access=protected)
        function setSensingMatrixSqr(obj, num_rows, num_columns, order, PL, PR)
            if nargin < 3
                sm_args = {};
            else
                sm_args = {num_rows, num_columns};
            end
            obj.setSensingMatrix(sm_args{:});
            
            if nargin >= 6
                obj.setPermutations(order, PL, PR);
            end
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
    end

    
end
