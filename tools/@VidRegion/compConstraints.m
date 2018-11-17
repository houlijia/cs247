function cnstrnts = compConstraints(obj,stt)
  xpnd = obj.getExpandMtrx(VidBlocker.BLK_STT_RAW, stt);
  unxpnd = obj.getExpandMtrx(stt, VidBlocker.BLK_STT_RAW);
  mtxs = {xpnd.M, obj.expnd_pxl_mat, obj.unexpnd_pxl_mat, unxpnd.M};
  mtx = SensingMatrixCascade.constructCascade(mtxs);
  
  cnstrnts = SensingMatrixCombine([1,-1], ...
    {mtx, SensingMatrixUnit(mtx.nCols())});
end

