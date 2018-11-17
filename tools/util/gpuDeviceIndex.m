function idx = gpuDeviceIndex()
  g = gpuDevice();
  if ~isempty(g)
    idx = g.Index;
  else
    idx = 0;
  end
end