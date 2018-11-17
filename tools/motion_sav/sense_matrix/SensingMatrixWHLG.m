classdef SensingMatrixWHLG < SensingMatrixWH
    %SensingMatrixWHLG is like SensingMatrixWH, but prerandomization is done
    %locally by multiplying by Bernoulli random variables, and then globally.
    %by premutation
    
    properties
        sign_changer = [];
    end
    
    methods
        % Constructor
        %   Input:
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        function obj = SensingMatrixWHLG(varargin)
            obj.set(varargin{:})
        end
        
        % Set Initialize
        %   Input:
        %     obj - this object
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
       function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixWH(varargin{:});
        end
        
        function setOrder(obj, order, ~)
            obj.setOrder@SensingMatrixWH(order)
            obj.setSignChanger();
        end
                
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
            
            x(obj.sign_changer) = - x(obj.sign_changer);
            y = obj.doMultVec@SensingMatrixWH(x,m);
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
            chngr = obj.sign_changer;
            if n ~= length(obj.sign_changer)
                chngr = chngr(chngr <= n);
            end
            
            y = obj.doMultTrnspVec@SensingMatrixWH(x,n);
            y(chngr) = - y(chngr);
        end
        
        function n_no_clip=nNoClip(~)
            n_no_clip = 0;
        end
        
    end
    
    methods(Access=protected)
        function setSignChanger(obj)
            sq = obj.rnd_strm.randi([0,1],[obj.n_cols,1]);
            obj.sign_changer = find(sq);
        end
        
        function [PL, PR] = makePermutations(obj, order, ~)
            PL = obj.rnd_strm.randperm(order);
            PR = [obj.rnd_strm.randperm(obj.n_cols) (obj.n_cols+1:order)];
        end
  end
end

