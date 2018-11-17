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
    function obj=SensingMatrixRnd(varargin)
      % Constructor.  The arguments are optional
      % Input:
      %   0 argument:   default constructor
      %   1 argument: A struct with fields specifying the
      %     matrix. fields are:
      %       Fields used by setSensingMatrix()
      %       rnd_seed: (Optional) Seed of random number generator
      %       rnd_type: (Optional) Type of random number generator
      %     Other fields are ignored
      %   2 arguments: assumed to be num_rows and num_cols.
      %   3 arguments: assumed to be num_rows, num_cols, rnd_seed. If rnd_seed
      %     is a struct the expected fields are 'type' and 'seed'.
      %   4 arguments:  assumed to be num_rows, num_cols, rnd_seed and rnd_type.
      obj.setSensingMatrixRnd(varargin{:})
    end
    
    function set(obj, varargin)
      % Input:
      %   0 argument: default constructor
      %   1 argument: A struct with fields specifying the
      %     matrix. fields are:
      %       Fields used by setSensingMatrix()
      %       rnd_seed: (Optional) Seed of random number generator
      %       rnd_type: (Optional) Type of random number generator
      %     Other fields are ignored
      %   2 arguments: assumed to be num_rows and num_cols.
      %   3 arguments: assumed to be num_rows, num_cols, rnd_seed. If rnd_seed
      %     is a struct the expected fields are 'type' and 'seed'.
      %   4 arguments:  assumed to be num_rows, num_cols, rnd_seed and rnd_type.
      varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
        'rnd_seed', 'rnd_type'});
      obj.setSensingMatrixRnd(varargin{:});
    end
    
    function setSeed(obj, rnd_seed, rtype)
      if nargin >= 3
        obj.rnd_type = rtype;
      end
      obj.seed = double(rnd_seed);
      
      if ischar(obj.rnd_type)
        obj.rnd_strm = RandStream(obj.rnd_type, 'Seed', obj.seed);
      elseif ~obj.rnd_type
        obj.rnd_strm = RandStream(obj.default_rnd_type, 'Seed', obj.seed);
      else
        obj.rnd_strm = RndCStrm(obj.seed, obj.rnd_type-1);
      end
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
      % Input:
      %   1 argument: do nothing
      %   2 argument: seocnd argument is a struct with fields specifying the 
      %     matrix. fields are:
      %       Fields used by setSensingMatrix()
      %       rnd_seed: (Optional) Seed of random number generator
      %       rnd_type: (Optional) Type of random number generator
      %     Other fields are ignored
      %   3 arguments: assumed to be num_rows and num_cols.
      %   4 arguments: assumed to be num_rows, num_cols, rnd_seed. If rnd_seed
      %     is a struct the expected fields are 'type' and 'seed'.
      %   5 arguments:  assumed to be num_rows, num_cols, rnd_seed and rnd_type.

      switch nargin
        case 1
          return
        case 2
          opts = num_rows;
          if ~isfield(opts, 'rnd_seed')
            opts.rnd_seed = SensingMatrixRnd.default_seed;
          end
          if ~isfield(opts, 'rnd_type')
            opts.rnd_type = obj.rnd_type;
          end
          
        case 3
          opts = struct('num_rows', num_rows, 'num_cols', num_columns, ...
            'rnd_seed', SensingMatrixRnd.default_seed, ...
            'rnd_type', SensingMatrixRnd.default_rnd_type);
        case 4
          if isstruct(rnd_seed)
          opts = struct('num_rows', num_rows, 'num_cols', num_columns, ...
            'rnd_seed', rnd_seed.seed, 'rnd_type', rnd_seed.type);
          else
            opts = struct('num_rows', num_rows, 'num_cols', num_columns, ...
              'rnd_seed', rnd_seed, 'rnd_type', SensingMatrixRnd.default_rnd_type);
          end
        case 5
            opts = struct('num_rows', num_rows, 'num_cols', num_columns, ...
            'rnd_seed', rnd_seed, 'rnd_type', rnd_type);
      end
      
      obj.setSensingMatrix(opts);
      
      if isstruct(opts.rnd_seed)
        r_seed = opts.rnd_seed.seed;
        r_type = opts.rnd_seed.type;
      else
        r_seed = opts.rnd_seed;
        r_type = opts.rnd_type;
      end
      obj.setSeed(r_seed, r_type);
    end
    
  end
  
  methods (Static, Access=protected)
    function ign = ignoreInEqual()
      ign = [SensingMatrix.ignoreInEqual() {'rnd_strm'}];
    end
  end
end

