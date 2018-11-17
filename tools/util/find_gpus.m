function indcs = find_gpus()
  % find_gpus returns a list of the GPUS available for processing, skipping
  % the GPU used for display.
  % Presently it does it by verifying that the GPU name contains 'Tesla'
  % Output:
  %   indcs - an array of the indcs of the available GPUs (empty if none)
  
  mex_gpu =mexGPU_mex();
  if ~mex_gpu
    error('Looking for GPU while MEX SW is compiled for no GPU');
  end
  
  indcs = zeros(1,gpuDeviceCount());
  for k=1:length(indcs)
    g = gpuDevice(k);
    if regexp(g.Name, 'Tesla')
      indcs(k) = g.Index;
    end
  end
  
  indcs = indcs(indcs ~= 0);
  if isempty(indcs) && mex_gpu
    error('NO GPU found while MEX SW is compiled for GPU');
  end
end

