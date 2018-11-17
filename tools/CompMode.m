classdef CompMode < matlab.mixin.Copyable
  %CompMode determines whether GPU is used and whether single precision
  %  is used.
  
  properties
    use_gpu = [];  % If true, using GPU Device ([] = undefined)
    use_single = []; % If true, use single precision ([] = undefined)

    % pointer to cast function to the right accuracy and processor
    toFloat = @CompMode.toCPUDouble;
    toCPUFloat = @CompMode.toCPUDouble;
    toSInt = @CompMode.toCPUSInt;
    toIndex = @CompMode.toCPUIndex;
    toLogical = @CompMode.toCPULogical;
    
    % generate the right type of arrays
    zeros = @CompMode.zerosCPUDouble;
    ones = @CompMode.onesCPUDouble;
  end
  
  methods
    function obj = CompMode(use_g, use_s)
      % CompMode constructor. Can have 0 or 2 arguments
      if nargin == 0
        obj.use_gpu = CompMode.defaultUseGpu();
        obj.use_single = CompMode.defaultUseSingle();
      else
        obj.use_gpu = use_g;
        obj.use_single = use_s;
      end
    end
    
    function set.use_gpu(obj,val)
      if isempty(obj.use_gpu) || obj.use_gpu ~= logical(val)
        obj.use_gpu = logical(val);
        obj.setUseGpu(val);
      end
    end
    
    function set.use_single(obj,val)
      if isempty(obj.use_single) || obj.use_single ~= logical(val)
        obj.use_single = logical(val);
        obj.setUseSingle(val);
      end
    end
  end
  
  methods (Access = protected)
    function setUseGpu(obj,~)
      obj.setCastIndex();
      obj.setCastFloat();
    end
    
    function setUseSingle(obj,~)
      obj.setCastFloat();
    end
    
    function setCastIndex(obj)
      if obj.use_gpu
        obj.toIndex = @CompMode.toGPUIndex;
        obj.toSInt = @CompMode.toGPUSInt;
        obj.toLogical = @CompMode.toGPULogical;
      else
        obj.toIndex = @CompMode.toCPUIndex;
        obj.toSInt = @CompMode.toCPUSInt;
        obj.toLogical = @CompMode.toCPULogical;
      end
    end

    function setCastFloat(obj)
      if obj.use_gpu
        if obj.use_single
          obj.toFloat = @CompMode.toGPUSingle;
          obj.toCPUFloat = @CompMode.toCPUSingle;
          obj.zeros = @CompMode.zerosGPUSingle;
          obj.ones = @CompMode.onesGPUSingle;
        else
          obj.toFloat = @CompMode.toGPUDouble;
          obj.toCPUFloat = @CompMode.toCPUDouble;
          obj.zeros = @CompMode.zerosGPUDouble;
          obj.ones = @CompMode.onesGPUDouble;
        end
      else
        if obj.use_single
          obj.toFloat = @CompMode.toCPUSingle;
          obj.toCPUFloat = @CompMode.toCPUSingle;
          obj.zeros = @CompMode.zerosCPUSingle;
          obj.ones = @CompMode.onesCPUSingle;
        else
          obj.toFloat = @CompMode.toCPUDouble;
          obj.toCPUFloat = @CompMode.toCPUDouble;
          obj.zeros = @CompMode.zerosCPUDouble;
          obj.ones = @CompMode.onesCPUDouble;
        end
      end
    end
    
    function otr = copyElement(obj)
      otr = copyElement@matlab.mixin.Copyable(obj);
      otr.use_gpu = obj.use_gpu;
      otr.use_single = obj.use_single;
    end
  end
  
  
  methods(Static)
    function val = defaultUseGpu()
      persistent use__gpu__;
      if isempty(use__gpu__)
        use__gpu__ = mexGPU_mex();
      end
      val = use__gpu__;
    end
    
    function val = defaultUseSingle()
      global use__single__;
      if isempty(use__single__)
        use__single__ = false;
      end
      val = use__single__;
    end
    
    function setDefaultUseSingle(val)
      global use__single__;
      use__single__ = val;
    end
     
    % cast functions
    function x = toCPUDouble(x)
      if isa(x, 'gpuArray')
        x = gather(x);
      end
      x = double(x);
    end
    
    function x = toCPUSingle(x)
      if isa(x, 'gpuArray')
        x = gather(x);
      end
      if ~issparse(x)
        x = single(x);
      else
        x = single(full(x));
      end
    end
    
    function x = toGPUDouble(x)
      x = double(gpuArray(full(x)));
    end
    
    function x = toGPUSingle(x)
      x = gpuArray(single(full(x)));
    end
    
    function x = toCPUIndex(x)
      if isa(x, 'gpuArray')
        x = gather(x);
      end
      x = uint32(full(x));
    end

    function x = toGPUIndex(x)
      x = gpuArray(uint32(full(x)));
    end
    
    function x = toCPUSInt(x)
      if isa(x, 'gpuArray')
        x = gather(x);
      end
      x = int32(full(x));
    end

    function x = toGPUSInt(x)
      x = gpuArray(int32(full(x)));
    end
    
    function x = toCPULogical(x)
      if isa(x, 'gpuArray')
        x = gather(x);
      end
      x = logical(full(x));
    end

    function x = toGPULogical(x)
      x = gpuArray(logical(full(x)));
    end
    
    % zero and one functions
    function v = zerosCPUDouble(varargin)
      args = [{'zeros'}, varargin];
      v = builtin(args{:});
    end

    function v = zerosCPUSingle(varargin)
      args = [{'zeros'}, varargin, {'single'}];
      v = builtin(args{:});
    end
    
    function v = zerosGPUDouble(varargin)
      args = [{'zeros'}, varargin, {'gpuArray'}];
      v = builtin(args{:});
    end

    function v = zerosGPUSingle(varargin)
      args = [{'zeros'}, varargin, {'single', 'gpuArray'}];
      v = builtin(args{:});
    end
    
    function v = onesCPUDouble(varargin)
      args = [{'ones'}, varargin];
      v = builtin(args{:});
    end

    function v = onesCPUSingle(varargin)
      args = [{'ones'}, varargin, {'single'}];
      v = builtin(args{:});
    end
    
    function v = onesGPUDouble(varargin)
      args = [{'ones'}, varargin, {'gpuArray'}];
      v = builtin(args{:});
    end

    function v = onesGPUSingle(varargin)
      args = [{'ones'}, varargin, {'single', 'gpuArray'}];
      v = builtin(args{:});
    end
        
  end
end

