% Documentation for this function is in VidRegion.m next to the function 
% signature
function [xcor, blk_motion] = nextFrmXCor(obj, vec, blk_stt, opts)
  if nargin < 4
    opts = CS_AnlsParams();
  end
  
  if opts.nrm_exp == -1
    compare = @(pxl_vec, vec_indcs, offsets) obj.compXCor(pxl_vec, vec_indcs, offsets);
  else
    compare = @(pxl_vec, vec_indcs, offsets) obj.compXDiff(pxl_vec, vec_indcs, offsets,...
      opts.nrm_exp);
  end
  
  if iscell(vec)
    src_vec = obj.vectorize(vec);
  else
    src_vec = vec;
  end
  
% Perform edge detection
  if any(opts.edge_rng)
    nghbr_ofsts = get_neighbor_ofsts(opts.edge_rng); % including zero offset
    vec_indcs = obj.inRngVec(nghbr_ofsts, blk_stt);
    nghbr_list = obj.offsetPxlToVec(nghbr_ofsts, false);
    vec = zeros(size(src_vec));
    for k=1:length(nghbr_list)
      vec(vec_indcs) = vec(vec_indcs) + src_vec(vec_indcs+nghbr_list(k));
    end
    vec(vec_indcs) = src_vec(vec_indcs) - (1./length(nghbr_list)*vec(vec_indcs));
  else
    nghbr_ofsts = [];
  end

  xcor = BlkMotnData(opts);
  orig = SimpleFractions([0 0], 1); % Initial search origin
  for k=1:xcor.nStages()
    % Generate the offsets of the grid for search (not including  origin
    offsets = xcor.compOffsets(k,orig);
    if opts.fxd_trgt
      if ~isempty(nghbr_ofsts)
        [needed_ofsts, ~, ~] = find_edge_ofsts(nghbr_ofsts, [0,0,0; offsets]);
      else
        needed_ofsts = [0,0,0; offsets];
      end
      vec_indcs = obj.inRngVec(rat(needed_ofsts, blk_stt));
      mtch = compare(vec, vec_indcs, rat(offsets));
    else
      mtch = zeros(size(offsets,1),1);
      for j=1:size(offsets,1)
        ofsts_j = offsets(j,:);
        if ~isempty(nghbr_ofsts)
          [needed_ofsts, ~, ~] = find_edge_ofsts(nghbr_ofsts, [0,0,0; ofsts_j]);
        else
          needed_ofsts = [0,0,0; ofsts_j];
        end
        vec_indcs = obj.inRngVec(rat(needed_ofsts), blk_stt);
        mtch(j) = obj.compare(vec, vec_indcs, rat(ofsts_j));
      end
    end
    mtch_info = xcor.compMax(mtch);
    xcor.setStageData(k, mtch_info);
    orig = offsets(mtch_info.mx_ind,1:2) / offsets(mtch_info.mx_ind,3);
  end
  blk_motion = xcor.next_frm_corr_sort();
end

