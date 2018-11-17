function xpnd = getExpandMtrx(obj, inp_stt, out_stt, eps)
  if nargin < 4
    eps = 0;
  end
  
  xpblk = struct(...
    'M', cell(1,size(obj.blk_indx,1)),...
    'U', cell(1,size(obj.blk_indx,1)),...
    'L', cell(1,size(obj.blk_indx,1)),...
    'V', cell(1,size(obj.blk_indx,1)),...
    'I', cell(1,size(obj.blk_indx,1)),...
    'R', cell(1,size(obj.blk_indx,1)));
  for ib = 1:length(xpblk)
    xpblk(ib) = ...
      obj.blkr.getExpandMtrx(inp_stt, out_stt, obj.blk_indx(ib,:), eps);
  end
  
  if length(xpblk) == 1
    xpnd = xpblk;
  else
    % We convert xpblk to struct. For safety, we do not assume a specific
    % order of field names.
    xp_flds = fieldnames(xpblk);
    for k=1:length(xp_flds);
      xp_fld = xp_flds{k};
      switch xp_fld
        case 'M'
          M = k;
        case 'U'
          U = k;
        case 'L'
          L = k;
        case 'V'
          V = k;
      end
    end
    xpblk = struct2cell(xpblk);
    xpnd = struct('M',SensingMatrixBlkDiag.constructBlkDiag(xpblk(M,:)),...
      'U',[],'L',[],'V',[], 'I',[],'R',[]);
    [xpnd.L, xpnd.U, xpnd.V] = SensingMatrixBlkDiag.compSVDfromBlks(...
      xpblk(L,:)', xpblk(U,:)', xpblk(V,:)', eps);
    xpnd = VidBlocker.xpndSetNorms(xpnd);
    
    [LI,UI,VI] = SensingMatrix.invertSVD(xpnd.L, xpnd.U, xpnd.V, eps);
    xpnd.I = SensingMatrix.constructSVD(LI,UI,VI);
    xpnd.R = SensingMatrixCascade.constructCascade({UI,LI});
  end
end

