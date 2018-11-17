function lbl = quantMsr(msr, offset, intvl, n_bins)
  qm = ceil((offset + msr )/intvl);
  if(qm > n_bins || qm < 1)
    lbl = n_bins+1;
  else
    lbl = qm;
  end
end

