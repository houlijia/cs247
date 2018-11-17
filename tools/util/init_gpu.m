function init_gpu(gpu_indx)
  global current_gpu_index;

  if isempty(gpu_indx) || gpu_indx == 0
    % Disable GPU
    current_gpu_index = 0;
    if gpuDeviceCount()
      gpuDevice([]);
    end
  elseif ~mexGPU_mex()
    error('GPU initialize attempt while MEX SW is compiled for no GPU');
  elseif ~isequal(gpu_indx, current_gpu_index)
    % Enable GPU or change GPU
    gpuDevice([]);
    current_gpu_index = cuda_init_mex(gpu_indx);
    if current_gpu_index > 0
      gpuDevice(current_gpu_index);
    else
      error('No GPU found');
    end
  end
end

