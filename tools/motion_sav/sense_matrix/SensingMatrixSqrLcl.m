classdef SensingMatrixSqrLcl < SensingMatrixSqr
    % SensingMatrixSqrLcl is the same as SensingMatrixSqr, except that
    % Pright is not a permutation matrix but a
    % diagonal matrix in which the diagonal elements get values of +-1.
    
    properties
    end
    
   
    methods
        % Constructor. Can have 1, 2 or 5 arguments or none
        function obj = SensingMatrixSqrLcl(varargin)
            obj.setSensingMatrixSqr(varargin{:});
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
                'order', 'PL', 'PR'});
            obj.setSensingMatrixSqr(varargin{:});
        end
        
        % doMultVec - implemenentation of abstract method of SensingMatrix -
        % multiply a vector x by A. The implementation uses the abstract
        % function multSqr().  Note that the first measurement is the sum
        % of all input vector entries and before multiplying we subtract
        % the mean from each input entry
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
            n = size(x,1);
            sm = sum(x);
            mn = sm/n;
            u = zeros(obj.sqr_order, size(x,2));
            for k=1:size(u,2)
                u(1:n,k) = obj.PR(1:n,1) .* (x(:,k)-mn(k));
            end
            v =obj.multSqr(u);
            y = [sm; v(obj.IPL(2:m),:)];
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
            m = size(x,1);
            u = zeros(obj.sqr_order, size(x,2));
            u(obj.IPL(1:(m-1)),:) = x(2:end,:);
            v =obj.multTrnspSqr(u);
            mn = min(n,length(obj.PR));
            y = v(1:n,:);
            for k=1:size(y,2)
                y(1:mn,k) = obj.PR(1:mn,1) .* y(1:mn,k);
            end
            y = y + obj.trnspScale()*x(1,:) - sum(y)/n;
            
        end
        
        function n_no_clip=nNoClip(~)
            n_no_clip = 1;
        end
        
        function dc_val = getDC(~,msrs)
            dc_val = msrs(1);
        end
    end

    
end
