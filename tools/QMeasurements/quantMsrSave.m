function [lbl,qm] = quantMsrSave(msr, offset, intvl, n_bins)
  qm = ceil((offset + msr )/intvl);
  if(qm > n_bins)
    lbl = n_bins+1;
    qm = qm - n_bins;
  elseif(qm < 1)
    lbl = n_bins+1;
  else
   lbl = qm;
  end
end

