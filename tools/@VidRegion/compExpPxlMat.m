function compExpPxlMat(obj)
  clr_mtx = cell(obj.n_color,1);
  blk_mtx = cell(obj.n_color, obj.n_blk);
  d_mtx = cell(1,3);
  clr_len = zeros(obj.n_color,1);
  
  for iblk = 1:obj.n_blk
    [origin, orig_end, ~,~,~,~] = ...
      obj.blkr.blkPosition(obj.blk_indx(iblk,:));
    origin = origin - obj.ofst_pxl_cnt_map;
    orig_end = orig_end - obj.ofst_pxl_cnt_map;
    
    for iclr = 1:obj.n_color
      for dim = 1:3
        d_mtx{dim} = SensingMatrixSelectRange.constructSelectRange(...
          origin(iclr,dim), orig_end(iclr,dim), ...
          size(obj.pxl_cnt_map{iclr},dim), false);
      end
      blk_mtx{iclr, iblk} = SensingMatrixKron.constructKron(d_mtx(end:-1:1));
    end
  end

  for iclr = 1:obj.n_color
    clr_len(iclr) = numel(obj.pxl_cnt_map{iclr});
    indcs = find(obj.pxl_cnt_map{iclr}(:));
    clr_mtx{iclr} = build_clr_mtx(indcs);
  end   
  clr_end = cumsum(clr_len);
  clr_bgn = 1 + [0; clr_end(1:end-1)];
  for iclr = 1:obj.n_color
    slct_clr_mtx = SensingMatrixSelectRange.constructSelectRange(...
      clr_bgn(iclr), clr_end(iclr), clr_end(obj.n_color));
    for iblk = 1:obj.n_blk
      blk_mtx{iclr, iblk} = ...
        SensingMatrixCascade({blk_mtx{iclr, iblk}, slct_clr_mtx});
    end
  end
  
  obj.expnd_pxl_mat = SensingMatrixCascade.constructCascade({...
    SensingMatrixConcat.constructConcat(blk_mtx(:)),...
    SensingMatrixBlkDiag.constructBlkDiag(clr_mtx)});
    
  % This nested function was added to avoid a warning about usage of find.
  function mt = build_clr_mtx(indcs)
    mt = SensingMatrixSelect.construct(indcs, obj.n_clr_pxls(iclr), true);
  end
end

