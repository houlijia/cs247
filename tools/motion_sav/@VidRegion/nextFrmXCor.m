% Documentation for this function is in VidRegion.m next to the function 
% signature
function [xcor, blk_motion] = nextFrmXCor(obj, vec, opts)
  if nargin < 3
    opts = CS_AnlsParams();
  end
  
  if opts.nrm_exp == -1
    compare = @(pxl_vec, offsets) obj.compXCor(pxl_vec, offsets);
  else
    compare = @(pxl_vec, offsets) obj.compXDiff(pxl_vec, offsets, opts.nrm_exp);
  end
  
  xcor = BlkMotnData(opts);
  orig = SimpleFractions([0 0], 1); % Initial search origin
  for k=1:xcor.nStages()
    % Generate the offsets of the grid for search (not including  origin
    offsets = xcor.compOffsets(k,orig);
    ofsts_3d = rat(offsets);
    if opts.fxd_trgt
      mtch = compare(vec, ofsts_3d);
    else
      mtch = zeros(size(offsets,1),1);
      for j=1:size(offsets,1)
        mtch(j) = obj.compXCor(vec, ofsts_3d(j,:));
      end
    end
    mtch_info = xcor.compMax(mtch);
    xcor.setStageData(k, mtch_info);
    orig = offsets(mtch_info.mx_ind,1:2) / offsets(mtch_info.mx_ind,3);
  end
  blk_motion = xcor.next_frm_corr_sort();
end

