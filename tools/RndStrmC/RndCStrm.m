classdef RndCStrm < RandStream
    %RandStream is a class which gives the same functionality as RandStream
    %with the Mersenne Twister generator (mt19937ar), but using mex
    %functions.
    
    properties (Access=private)
        state;
        tst = false;
    end
    
    methods
        % Constructor
        %   Input:
        %     seed - generator seed (default=0)
        %     cmpr - If true compare mex functions to RandStream functions
        %            (default = false)
        function obj = RndCStrm(seed, cmpr)
            if nargin < 2
                cmpr = false;
                if nargin < 1
                    seed = false;
                end
            end
            
            obj = obj@RandStream('mt19937ar', 'Seed', seed);
            obj.state = init_RndC_mex(seed);
            obj.tst = cmpr;
        end
        
        function r = rand(varargin)
            obj = varargin{1};
            varargin = varargin(2:end);
            args = varargin;
            single_out = false;
            if ~isempty(args) && ischar(args{end})
                switch args{end}
                    case 'double'
                        single_out = false;
                    case 'single'
                        single_out = true;
                    otherwise
                        error('unexpected precision: %s', args{end});
                end
                args = args(1:end-1);
            end
            
            [sz,cnt] = obj.get_size(args{:});

            [obj.state, r] = rand_RndC_mex(obj.state, cnt);
            
            if single_out
                r = single(r);
            end
            r = reshape(r, sz);
            
            if obj.tst
                ref = obj.rand@RandStream(varargin{:});
                if ~isequal(ref,r)
                    error('rand: mex and RandStream results not the same');
                end
            end
        end
        
        function r = randi(varargin)
            obj = varargin{1};
            varargin = varargin(2:end);
            args = varargin;
            if ischar(args{end})
                classname = args{end};
                args=args(1:end-1);
            else
                classname = 'double';
            end
            
            imax = args{1};
            args = args(2:end);
            if length(imax) == 1
                offset = 0;
            else
                offset = imax(1)-1;
                imax = imax(2) - (imax(1)-1);
            end
            if imax > intmax('uint32')
                scaler = double(imax) / double(intmax('uint32'));
                imax = intmax('uint32');
            else
                scaler = 1;
            end
            
            [sz,cnt] = obj.get_size(args{:});

            [obj.state, r] = randi_RndC_mex(obj.state, imax, cnt);
            if scaler ~= 1
                r = scaler * double(r);
            end
            r = cast(r,classname);
            r = r+offset;
            
            r = reshape(r, sz);
            
            if obj.tst
                ref = obj.randi@RandStream(varargin{:});
                if ~isequal(ref,r) && ...
                        (scaler == 1 || norm((r(:)-ref(:)),inf) >= scaler)
                    error('randi: mex and RandStream results not the same');
                end
            end
        end
        
        function r = randn(varargin)
            obj = varargin{1};
            varargin = varargin(2:end);
            args = varargin;
            single_out = false;
            if ~isempty(args) && ischar(args{end})
                switch args{end}
                    case 'double'
                        single_out = false;
                    case 'single'
                        single_out = true;
                    otherwise
                        error('unexpected precision: %s', args{end});
                end
                args = args(1:end-1);
            end
            
            [sz,cnt] = obj.get_size(args{:});

            [obj.state, r] = randn_RndC_mex(obj.state, cnt);
            
            if single_out
                r = single(r);
            end
            r = reshape(r, sz);
            
            if obj.tst
                ref = obj.randn@RandStream(varargin{:});
                if ~isequal(ref,r)
                    error('rand: mex and RandStream results not the same');
                end
            end
        end
        
        function r = randperm(varargin)
            obj = varargin{1};
            imax = varargin{2};
            if nargin < 3
                [obj.state, r] = randperm_RndC_mex(obj.state, imax);
            else
                cnt = varargin{3};
                if cnt> imax
                    error('randperm(imax=%d,cnt=%d) - cnt cannot be > imax', cnt, imax);
                else
                    [obj.state, r] = randperm_RndC_mex(obj.state, imax, cnt);
                end
            end      
            if obj.tst
                if nargin < 3
                    ref = obj.randperm@RandStream(imax);
                else
                    ref = obj.randperm@RandStream(imax,cnt);
                end
                if ~isequal(ref,r)
                    error('randperm: mex and RandStream results not the same');
                end
            end
        end
        
    end
    
    methods (Static, Access=private)
        function [sz,cnt] = get_size(varargin)
            if isempty(varargin)
                sz = [1,1];
            elseif length(varargin) == 1
                d = varargin{1};
                if length(d) == 1
                    sz = [d,d];
                else
                    sz = d;
                end
            else
                sz = ones(1,length(varargin));
                for k=1:length(sz)
                    sz(k) = varargin{k};
                end
            end
            cnt = prod(sz);
        end
    end
end

