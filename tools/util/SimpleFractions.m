classdef SimpleFractions
    %SimpleFractions represents arrays of rational numbers which have the
    %same denominator;
    
    properties (Access = protected);
        numer;
        denom;
    end
    
    methods
        function obj = SimpleFractions(nmr, dnm)
            if nargin < 2
                dnm = 1;
                if nargin==1 && isa(nmr,'SimpleFractions') && ...
                        builtin('numel',nmr)==1
                    obj.numer = nmr.numer;
                    obj.denom = nmr.denom;
                    return
                elseif nargin <1
                    nmr = 0;
                end
            end
            
            if dnm < 0
                nmr = -nmr;
                dnm = -dnm;
            end
            if ~isequal(floor([nmr(:);dnm]),[nmr(:);dnm])
                q = nmr/dnm;
                [N,D] = rat(q, 1e-9*norm(q(:),1));
                cmn_dnm = obj.lcm(D);
                nmr = N .* ((cmn_dnm * ones(size(D))) ./ D);
                dnm = cmn_dnm;
            end
            
            obj.numer = nmr;
            obj.denom = dnm;
        end
        
        function [nmr,dnm] = rat(obj)
            nmr = obj.numer;
            dnm = obj.denom;
        end
        
        function str = show_str(obj, fmt, ~)
            if nargin < 2
                fmt = struct();
            end
            numer_str = show_str(obj.numer, fmt);
            denom_str = show_str(obj.denom, fmt);
            str = sprintf('%s<[%s]/%s>', class(obj),numer_str, denom_str);
%             str = sprintf('%s<[%s]/%s>', ...
%                 class(obj),show_str(obj.numer, fmt), show_str(obj.denom, fmt));
        end
        
        function disp(obj)
            disp(show_str(obj));
        end
        
        % normalize
        function a = normalize(a)
            g = SimpleFractions.gcd([a.numer(:);a.denom]);
            a.numer = a.numer /g;
            a.denom = a.denom /g;
        end
        
        % cast to double
        function val = double(obj)
            val = obj.numer / obj.denom;
        end
        
        % size, length, numel, end operators
        function sz = size(a, dim)
            if nargin == 1
                sz = size(a.numer);
            else
                sz = size(a.numer,dim);
            end
        end
        function ln = length(a)
            ln = length(a.numer);
        end
        function nm = numel(a)
            nm = numel(a.numer);
        end
        function out = end(a,dim,ndim)
            if ndim==1
                out = numel(a.numer);
            else
                out = size(a.numer,dim);
            end
        end
        
        % unary sign operation
        function a = uminus(a)
            a.numer = -a.numer;
        end
        function a = uplus(a)
            a.numer = a.numer;
        end
        
        % plus and minus
        function val = plus(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            [a,b] = SimpleFractions.normalize2(a,b);
            val = SimpleFractions(a.numer+b.numer, a.denom);
        end
        function val = minus(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            [a,b] = SimpleFractions.normalize2(a,b);
            val = SimpleFractions(a.numer-b.numer, a.denom);
        end
        
        % element-wise multiply .*
        function val = times(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = SimpleFractions(a.numer .* b.numer, a.denom * b.denom);
            val = val.normalize();
        end
        
        % matrix multiply *
        function val = mtimes(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = SimpleFractions(a.numer * b.numer, a.denom * b.denom);
            val = val.normalize();
        end
        
        % right element-wise divide: ./
        function val = rdivide(a,b)
          [a,b] = SimpleFractions.import2(a,b);
          [a,b] = SimpleFractions.normalize2(a,b);
          d = SimpleFractions.lcm(b.numer);
          val = SimpleFractions(a.numer .* ((d*b.denom) ./ b.numer), d*a.denom);
          val = normalize(val);
        end
        
        % right matrix divide: /
        function val = mrdivide(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            [a,b] = SimpleFractions.normalize2(a,b);
            val = SimpleFractions(a.numer / b.numer);
        end
        
        % Comparison operators
        function val = lt(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = (a.numer * b.denom < b.numer * a.denom);
        end
        function val = gt(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = (a.numer * b.denom > b.numer * a.denom);
        end
        function val = le(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = (a.numer * b.denom <= b.numer * a.denom);
        end
        function val = ge(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = (a.numer * b.denom >= b.numer * a.denom);
        end
        function val = ne(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = (a.numer * b.denom ~= b.numer * a.denom);
        end
        function val = eq(a,b)
            [a,b] = SimpleFractions.import2(a,b);
            val = (a.numer * b.denom == b.numer * a.denom);
        end
        
        % min and max
        function val = min(a,b)
            if nargin == 1
                if isempty(a)
                    val = a;
                else
                    val = SimpleFractions(min(a.numer), a.denom);
                end
                return
            end
            [a,b] = SimpleFractions.import2(a,b);
            [a,b] = SimpleFractions.normalize2(a,b);
            val = a;
            indx = find(a.numer > b.numer);
            val.numer(indx) = b.numer(indx);
            val = normalize(val);
        end
        function val = max(a,b)
            if nargin == 1
                if isempty(a)
                    val = a;
                else
                    val = SimpleFractions(max(a.numer), a.denom);
                end
                return
            end
            [a,b] = SimpleFractions.import2(a,b);
            [a,b] = SimpleFractions.normalize2(a,b);
            val = a;
            indx = find(a.numer < b.numer);
            val.numer(indx) = b.numer(indx);
            val = normalize(val);
        end
        
        % floor, ceil, fix and round (return value is double)
        function val = floor(a)
            val = floor(a.numer/a.denom);
        end
        function val = ceil(a)
            val = ceil(a.numer/a.denom);
        end
        function val = fix(a)
            val = fix(a.numer/a.denom);
        end
        function val = round(a)
            val = round(a.numer/a.denom);
        end
       
        % Returns a logical array of the same size, with true for entries
        % which are integers
        function val = isInt(a)
          val = (mod(a.numer,a.denom) == 0);
        end
        
        % Conjugate (complex)
        function val = ctranspose(a)
            if ~isa(a,'SimpleFractions')
                a = SimpleFractions(a);
            end
            val = SimpleFractions(a.numer', a.denom);
        end
            
        % concatenation
        function val = horzcat(varargin)
            val = SimpleFractions(varargin{1});
            for k=2:nargin
                b = SimpleFractions(varargin{k});
                [val,b] = SimpleFractions.normalize2(val,b);
                val.numer = [val.numer, b.numer];
            end
        end
        function val = vertcat(varargin)
            val = SimpleFractions(varargin{1});
            for k=2:nargin
                b = SimpleFractions(varargin{k});
                [val,b] = SimpleFractions.normalize2(val,b);
                val.numer = [val.numer; b.numer];
            end
        end
        
        % Subscripted reference
        function val = subsref(obj,ref)
            if length(ref) > 1
                val = subsref(subsref(obj,ref(1)), ref(2:end));
            else
                switch ref.type
                    case {'()','{}'}
                        val = SimpleFractions(subsref(obj.numer,ref), obj.denom);
                    case '.'
                        val = obj.(ref.subs);
                end
            end
        end
        
        function obj = subsasgn(obj, ref, val)
          switch ref.type
            case {'()','{}'}
              [obj,val] = SimpleFractions.import2(obj,val);
              [obj,val] = SimpleFractions.normalize2(obj,val);
              obj.numer = subsasgn(obj.numer, ref, val.numer);
              obj = normalize(obj);
            case '.'
              obj = subsasgn(obj, ref, val);
          end
        end
    end
    
    methods (Access = protected)
    end
   
    methods (Static)
        function val = lcm(arr)
            arr = abs(arr(arr~=0));
            d = SimpleFractions.gcd(arr);
            val = d * prod(arr(:)/d);
        end
        
        function val = gcd(arr)
            arr = arr(arr~=0);
            val = arr(1);
            for k=2:numel(arr)
                val = gcd(val, arr(k));
                if val == 1
                    break
                end
            end
        end
    end
    
    methods(Static, Access=protected)
      function [a,b] = import2(a,b)
        if ~isa(a,'SimpleFractions')
          a = SimpleFractions(a);
        end
        if ~isa(b,'SimpleFractions')
          b = SimpleFractions(b);
        end
      end
        
       function [a,b] = normalize2(a,b)
           % Bring to common denominator
           g = SimpleFractions.gcd([a.numer(:);a.denom;b.numer(:);b.denom]);
           ad = a.denom / g;
           bd = b.denom /g;
           a.numer = (a.numer / g)*bd;
           a.denom = ad * bd;
           b.numer = (b.numer / g)*ad;
           b.denom = a.denom;
        end
    end           
end

