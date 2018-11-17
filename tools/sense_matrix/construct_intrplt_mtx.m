function intrpl_mtx = construct_intrplt_mtx(sz_in, sz_out)
  % construct_intrplt_mtx creates a matrix for interpolating an signal of
  % size sz_in into a signal of size sz_out.
  %  
  %   Input arguments:
  %     sz_in - size of the input signal. May be a scalar, if the input is
  %             one-dimensional or a row vector for multidimensional
  %             signals.
  %     sz_out - size of output signal. should be of the same size as sz_in
  %   Output:
  %     intrpl_mtx - an object of type SensingMatrix which can perform the
  %                  interpolation.
  %   If x is a vector, the linearly interpolated vector is obtained by
  %      y=intrpl_mtx.multVec(x).
  %   If x is a multi-dimesnional signal such that sz_in = size(x) and
  %      y=reshape(intrpl_mtx.multVec(x(:)), sz_out)
  %   then size(y) = sz_out and y is the linearly interpolated version of
  %   x.
  %
  %   if m2=construct_intplt_mtx(sz1, sz2) and 
  %      m1=construct_intplt_mtx(sz2, sz1)
  %   then m1*m2 is a unit matrix if each element in sz2 is divisble by the
  %   corresponding element in sz1.
  
  if ~(all(size(sz_in)==size(sz_out)) && isrow(sz_in) && ...
      isnumeric(sz_in) && isnumeric(sz_out) && ...
      all(sz_in>0) && all(sz_out>0) && ...
      ~any(mod(sz_in,1)) && ~any(mod(sz_out,1)))
    error(...
      'sz_in and sz_out should be row vectors positive integers of same dimension');
  end

  nd = length(sz_in);
  if nd > 1
    mtrcs = cell(nd);
    for k=1:nd
      mtrcs{k} = construct_intrplt_mtx(sz_in(k), sz_out(k));
    end
    intrpl_mtx = SensingMatrixKron.constructKron(mtrcs(nd:-1:1));
  elseif sz_out == 1
    intrpl_mtx = SensingMatrixSelect.construct(1,sz_in,false);
  elseif sz_in == 1
    intrpl_mtx = SensingMatrixDC.constructDC(sz_out, true, false);
  elseif ~mod(sz_in, sz_out) % sz_out divides sz_in - simple selection
    intrpl_mtx = SensingMatrixKron.constructKron({...
      SensingMatrixUnit(sz_out), SensingMatrixSelect.construct(1,sz_in/sz_out)});
  else
    nc = gcd(sz_in, sz_out);
    n_in = sz_in/nc;
    n_out = sz_out/nc;
    pos = (0:n_out-1)*(n_in/n_out);
    int_pos = floor(pos);
    wgt = pos - int_pos;
    sl = SensingMatrixSelect.construct(1+int_pos, n_in);
    r0 = SensingMatrixCascade.constructCascade({SensingMatrixDiag(1-wgt), sl});
    r1 = SensingMatrixCascade.constructCascade({SensingMatrixDiag(wgt), sl});
    rpt = SensingMatrixUnit(nc);
    s2 = SensingMatrixConcat({...
      SensingMatrixSelectRange(2,sz_in,sz_in), ...
      SensingMatrixSelect(sz_in, sz_in)});
    c0 = SensingMatrixKron.constructKron({rpt,r0});
    c1 = SensingMatrixCascade.constructCascade({...
      SensingMatrixKron.constructKron({rpt,r1}), s2});
    intrpl_mtx = SensingMatrixCombine([1 1], {c0,c1});
  end
  
end

