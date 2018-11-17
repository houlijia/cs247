function compUnExpPxlMat(obj)
  mtx = obj.expnd_pxl_mat.copy();
  mtx.transpose();
  wgt = 1 ./ mtx.multVec(mtx.ones(mtx.nCols(),1));
  obj.unexpnd_pxl_mat = SensingMatrixCascade.constructCascade({...
    SensingMatrixDiag.constructDiag(wgt), mtx});
end

