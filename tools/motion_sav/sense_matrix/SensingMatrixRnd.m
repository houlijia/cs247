classdef SensingMatrixRnd < SensingMatrix
    %A sensing matrix with some randomality in it.
    %   Detailed explanation goes here
    
    properties
        seed = SensingMatrixRnd.default_seed;
        
        % rnd_type is the type of random number generator used. It can be
        % either a string or one of the numbers 0,1,2. If it is a string,
        % this is the RandStream type. If it is zero, the default type is
        % used (default_rnd_type). If it non-zero we use RndCStrm instead
        % of RandStream. In that case, a value of 1 indicate no testing
        % (comparing with RandStream) while 2 indicates testing.
        rnd_type = 1;
        rnd_strm;
    end
    
    properties (Constant)
        default_seed = 0;
        default_rnd_type = 'mlfg6331_64';
    end
    
    methods
        % Constructor.  The argument are optional
        function obj=SensingMatrixRnd(varargin)
            obj.setSensingMatrixRnd(varargin{:})
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
                'rnd_seed', 'rnd_type'});
            obj.setSensingMatrixRnd(varargin{:});
        end
        
        function eql = isEqual(obj,other)
            if class(obj) ~= class(other)
                eql = false;
                return;
            end
            
            otr = other.copy();
            otr.code = obj.code;
            otr.rnd_strm  = obj.rnd_strm;
            
            eql = isequal(obj, otr);
        end
        
        function setSeed(obj, rnd_seed, rtype)
            if nargin >= 3
                obj.rnd_type = rtype;
            end
            obj.seed = rnd_seed;
            
            if ischar(obj.rnd_type)
                obj.rnd_strm = RandStream(obj.rnd_type, 'Seed', obj.seed);
            elseif ~obj.rnd_type
                obj.rnd_strm = RandStream(obj.default_rnd_type, 'Seed', obj.seed);
            else
                obj.rnd_strm = RndCStrm(obj.seed, obj.rnd_type-1);
            end
        end
        
        % encode - write out the randomization properties
        function len=encode(obj, code_dst, info)
            len = obj.encode@SensingMatrix(code_dst, info);
            if ischar(len)
                return;
            end
            total = len;
                
            len = obj.encodeRnd(code_dst, info);
            if ischar(len)
                return;
            end
            len = total + len;
        end
        
        % decode - read the randomization properties
        function len=decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            len = obj.decode@SensingMatrix(code_src, info, cnt);
            if ischar(len)
                return
            end
            total = len;
            
            len = obj.decodeRnd(code_src, info, cnt-total);
            if ischar(len)
                return;
            end
            len = len+total;
        end
    end
    
    methods (Static)
        % Get the properties of a random stream in a standardized form.
        % "Standardized" means that the State field is represented by two
        % uint32 arrays.
        % Input:
        %   props - output of RandStream.get()
        % Output
        %   props - standardized properties
        function props = stdRndStrmProps(props)
            state = props.State;
            switch class(state)
                case 'uint32'
                    state_h = state;
                    state_l = [];
                case 'int32'
                    state_h = SensingMatrixRnd.int32_to_uint32(h);
                    state_l = [];
                case 'uint64'
                    s_h = bitshift(state, -32);
                    state_h = uint32(s_h);
                    state_l = uint32(state - bitshift(s_h, 32));
                case 'int64'
                    s_h = bitshift(state, -32);
                    state_h = SensingMatrixRnd.int32_to_uint32(int32(s_h));
                    state_l = uint32(state - bitshift(s_h, 32));
                otherwise
                    error('Unexpected state type: %s', class(state));
            end
            
            props = rmfield(props,'State');
            props.State_h = state_h;
            props.State_l = state_l;
        end
        
        % Set standardized properties, received by getStdRndStrmProps()
        % into a stream
        %   Input
        %     state_class - (string) class of State property
        %     props - stadardized properties
        %   Output
        %     props - the modidified standardized properties
        function props = unStdStrmProps(state_class, props)
            switch state_class
                case 'uint32'
                    props.State = props.state_h;
                case 'int32'
                    props.State = ...
                        SensingMatrixRnd.uint32_to_int32(props.state_h);
                case 'uint64'
                    props.State = bitshift(uint64(props.state_h),32)+ ...
                        uint64(props.state_l);
                case 'int64'
                    state_h = SensingMatrixRnd.uint32_to_int32(props.state_h);
                    props.State = bitshift(int64(state_h),32)+ ...
                        int64(props.state_l);
                otherwise
                    error('Unexpected state type: %s', class(state));
            end
            props = rmfield(props,'state_l');
            props = rmfield(props,'state_h');
        end
        
        function u = int32_to_uint32(s)
            u = uint32(s);
            neg = find(s<0);
            u(neg) = uint32(int64(2^32) + int64(s(neg)));
        end
        
        function s = uint32_to_int32(u)
            neg = find(u > uint32(intmax('int32')));
            s = int32(u);
            s(neg) = int32(int64(u(neg))-int64(2^32));
        end
    end
        
    methods (Access=protected)
        function setSensingMatrixRnd(obj, num_rows, num_columns, ...
                rnd_seed, rnd_type)
            if nargin < 5
              if nargin==4 && isstruct(rnd_seed)
                rnd_type = rnd_seed.type;
                rnd_seed = rnd_seed.seed;
              else
                rnd_type = SensingMatrixRnd.default_rnd_type;
              end
            end
            if nargin < 4
                rnd_seed = SensingMatrixRnd.default_seed;
            end
            if nargin < 3
                sm_args = {};
            else
                sm_args = {num_rows, num_columns};
            end
            obj.setSensingMatrix(sm_args{:});
            
            obj.setSeed(rnd_seed, rnd_type);
        end
            
        % encodeRnd - write out the randomization properties
        function len=encodeRnd(obj, code_dst, ~)
            len = code_dst.writeUInt(obj.seed);
            if ischar(len)
                return;
            end
            
            if ischar(obj.rnd_type)
                rnd_str = obj.rnd_type;
            else
                rnd_str = int2str(obj.rnd_type);
            end
            len1 = code_dst.writeString(rnd_str);
            if ischar(len1)
                len = len1;
            else
                len = len + len1;
            end
        end
        
        % decodeRnd - read the randomization properties
        function len=decodeRnd(obj, code_src, ~, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            [val, len] = code_src.readUInt(cnt);
            if ischar(val) || val==-1
                len = val;
                return;
            end
            
            [type_str, len1] = code_src.readString(cnt-len);
            if ischar(len1)
                len = len1;
                return
            elseif len1 == -1
                len = 'EOD found';
                return
            end
            if regexp(type_str, '^\d+$')
                type = sscanf(type_str,'%d');
            else
                type = type_str;
            end
            
            obj.setSeed(val, type);
        end
    end
    
end

