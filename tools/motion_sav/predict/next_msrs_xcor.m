function [mcor, blk_motion, n_sum, vcor, vblk_motion] =...
  next_msrs_xcor(msrs, sens_mtrx, vid_region, opts, ref_vec)

  %next_msrs_xcor Computes normalized cross correlations between
  % measurements at offsets corresponding to one temporal frame shift and
  % various horizontal and vertical offsets
  %   Input:
  %     msrs - measurements vector
  %     sens_mtrx - the sensing matrix
  %     vid_region - Video region object
  %     opts - a CS_AnlsParams object
  %  Output:
  %    mcor - Normalized match score 3D array.  3rd index is negative
  %           temporal offset (-1,-2...).
  %    blk_motion - BlkMotion object describing the motion found in the
  %                 measurements
  %    n_sum - number of terms in the cross correlation estimate
  %    vcor  - computed only if opts.chk_ofsts is true and ref_vec is
  %            present. normalized cross correlation of shifted vectors.
  %    vblk_motion - BlkMotion object describing the motion found in the
  %                 shifted vectors.
  
  % Set defaults for input arguments
  if nargin <5
    ref_vec = [];
    if nargin < 4
      opts = CS_AnlsParams();
    end
  end
  if ~opts.chk_ofsts
    ref_vec = [];
  end
  
  if opts.nrm_exp == -1
    compare = @(v0,v1) comp_xcor(v0, v1);
    params = struct('nrm_exp',0);
  else
    compare = @(v0,v1) comp_xdiff(v0, v1, opts.nrm_exp);
    params = struct('nrm_exp',opts.nrm_exp);
  end
  
  nghbr_ofsts = get_neighbor_ofsts(opts.edge_rng); % including zero offset
  
  mcor = BlkMotnData(opts);
  orig = SimpleFractions([0 0], 1); % Initial search origin
  for k=1:mcor.nStages()
    % Generate the offsets of the grid for search (not including  origin)
    offsets = [0,0,0;mcor.compOffsets(k,orig)];
    
    if opts.fxd_trgt
      if ~isempty(nghbr_ofsts)
        [needed_ofsts, params.ofsts_list, params.nghbr_list] = ...
          find_edge_ofsts(nghbr_ofsts, offsets);
      end
      
      ofst_msrs = get_ofst_msrs(needed_ofsts, vid_region, ...
          sens_mtrx, msrs, params);
      n_sum = size(ofst_msrs,1);
      mtch = compare(ofst_msrs(:,1), ofst_msrs(:,2:end));
    else
      mtch = zeros(size(offsets,1),1);
      n_sum = zeros(size(offsets,1),1);
      
      for j=2:size(offsets,1)
        ofsts_j = [ offsets(1,:); offsets(j,:)];
        if ~isempty(nghbr_ofsts)
          [needed_ofsts, params.ofsts_list, params.nghbr_list] = ...
            find_edge_ofsts(nghbr_ofsts, ofsts_j);
        end
        ofst_msrs = get_ofst_msrs(needed_ofsts, vid_region, ...
          sens_mtrx, msrs, params);
        n_sum(j) = size(ofst_msrs,1);
        mtch(j) = compare(ofst_msrs(:,1), ofst_msrs(:,2:end));
      end
    end
    mtch_info = mcor.compMax(mtch);
    mcor.setStageData(k, mtch_info);
    % add 1 to skip the [0,0,0] entry in the beginning
    indx = mtch_info.mx_ind + 1;
    orig = offsets(indx,1:2) / offsets(indx,3);
  end
  blk_motion = mcor.next_frm_corr_sort();
  
  % Take care of ref_vec analysis only if necessary
  if isempty(ref_vec)
    return;
  end
  
  if length(nghbr_ofsts) > 1
    s_ref = CircShiftVec(ref_vec, ...
      vid_region.offsetPxlToVec(nghbr_ofsts(2:end,:),true));
    ref_vec = ref_vec - (1/s_ref.nCols()) * s_ref.addAll();
  end
  
  vcor = BlkMotnData(opts);
  orig = SimpleFractions([0 0], 1); % Initial search origin
  for k=1:vcor.nStages()
    % Generate the offsets of the grid for search (not including  origin
    offsets = vcor.compOffsets(k,orig);
    ofsts = rat(offsets);
    mtch = compare(ref_vec, ...
      CircShiftVec(ref_vec, vid_region.offsetPxlToVec(ofsts,true)));
    
    mtch_info = vcor.compMax(mtch);
    vcor.setStageData(k, mtch_info);
    orig = offsets(mtch_info.mx_ind,1:2) / offsets(mtch_info.mx_ind,3);
  end
  vblk_motion = vcor.next_frm_corr_sort();
  
  function ofst_msrs = get_ofst_msrs(needed_ofsts, vid_region, ...
          sens_mtrx, msrs, params)
    ofst_3d = rat(needed_ofsts);
    needed_ofs = vid_region.offsetPxlToVec(ofst_3d, true);
    ofst_msrs  = sens_mtrx.getOffsetMsrmnts(needed_ofs, msrs, ...
      'all', params);
%     ofst_msrs = needed_msrs(:,ofsts_list);
%     if ~isempty(nghbr_list) && ~isempty(ofst_msrs)
%       for jj=1:size(ofst_msrs,2)
%         ofst_msrs(:,jj) = ofst_msrs(:,jj) - (1/size(nghbr_list,2)) *...
%           sum(needed_msrs(:,nghbr_list(jj,:)'),2);
%       end
%     end
  end
end



