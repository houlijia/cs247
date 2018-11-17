% Get a matrix which extends the pixel vector by insterting zeros and
% doing pre-windowing if necessary.
% Input:
%   
function extnd_mtrx = getExtndMtrx(obj, sens_mtrx)
  % Computes a sensing matrix which peforms windowing and extends the pixel
  % vector by zero extension. If a second argument is included,
  % it is a sensing matrix which is cascaded with the extension
  % matrix.
  % Input:
  %   obj - this object
  %   sens_mtrx -  (optional) sensing matrix to extend
  
  wndm = obj.blkr.getWindowCBlkWgts();
  wndm = wndm(:);
  if ~all(wndm == 1)
    wndm = wndm * ones(1,obj.n_blk);
    wndm = wndm(:);
    wndm = wndm * ones(1,obj.n_color);
    wndm = wndm(:);
    lw = length(wndm);
    w_mtrx = SensingMatrixMatlab(sparse(1:lw,1:lw,wndm));
  else
    w_mtrx = [];
  end
  
  if any([obj.zext(:);obj.wext(:)])
    cbs = obj.clrBlkSize();
    zext_size = cbs + obj.zext(1,:) + ...
      obj.zext(2,:) + obj.wext;
    mtrcs = cell(1,3);
    for dim=1:3
      zdm = zext_size(dim);
      sdm = cbs(dim);
      ddm = zdm - sdm;
      if ddm == 0
        mtrcs{4-dim}=SensingMatrixUnit(zext_size(dim));
      elseif obj.wext(dim)
        mtrcs{4-dim} = SensingMatrixMatlab(sparse(...
          [(1:zdm) (sdm+1:zdm)], ...
          [(1:sdm) sdm*ones(1,ddm) ones(1,ddm)],...
          [ones(1,sdm) [(ddm:-1:1) (1:ddm)]/(ddm+1)]));
        
      else  % Zero extend
        mtrcs{4-dim} = SensingMatrixSelect(...
          (obj.zext(1,dim)+1):(zdm-obj.zext(2,dim)), zdm);
        mtrcs{4-dim}.transpose();
      end
    end
    k_mtrx = SensingMatrixKron(mtrcs);
    mtrcs = cell(obj.n_blk, obj.n_color);
    mtrcs(:) = {k_mtrx};
    if length(mtrcs(:)) == 1
      e_mtrx = mtrcs{1};
    else
      e_mtrx = SensingMatrixBlkDiag(mtrcs(:));
    end
  else
    e_mtrx =[];
  end
  
  mtrcs = cell(3:1);
  n_mtrcs = 0;
  
  if nargin >= 2
    n_mtrcs = n_mtrcs + 1;
    mtrcs{n_mtrcs} = sens_mtrx;
  end
  if ~isempty(e_mtrx);
    n_mtrcs = n_mtrcs + 1;
    mtrcs{n_mtrcs} = e_mtrx;
  end
  if ~isempty(w_mtrx);
    n_mtrcs = n_mtrcs + 1;
    mtrcs{n_mtrcs} = w_mtrx;
  end
  
  switch n_mtrcs
    case 0
      extnd_mtrx = [];
    case 1
      extnd_mtrx = mtrcs{1};
    otherwise
      extnd_mtrx = SensingMatrixCascade(mtrcs(1:n_mtrcs));
  end
  
end

