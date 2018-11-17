function sens_mtrx = genMsrmntMatrix(type, args, csr, nxt)
  % genMsrmntMatrix - Generate a measurement matrix
  % INPUT
  %   enc_opts: Encoding options object
  %   type: Type of the matrix (a string). The 'SensingMatrix' prefix may be
  %         omitted.
  %   args: Arguments for constructing the matrix. In particular, args is
  %         supposed to have the following fields:
  %     n_cols: Number of columns in the matrix.
  %     dims: Image dimensions
  %     rnd_seed: random number generator seed
  %   csr: The required overall compression ratio.
  %   nxt: (optional). If present and not empty, it is a struct which specifies
  %        another matrix, used to multiply the output of matrix specified above. 
  %        The fields of nxt should be 'type', 'args', csr' and optionally 'nxt',...
  %        thus it is a recursive definition. nxt.args does not need to have the
  %        fields n_cols, dims, process_color and rnd_seed, since they are
  %        propagated by this function.
  % OUTPUT
  %   sens_mtrx: The sensing matrix
  
  if nargin < 4
    nxt = [];
  end
  
  nr = args.n_cols * csr;
  if ~isempty(nxt)
    nr = nr / nxt.csr;
  end
  
  args.n_rows = round(nr);
  
  sens_mtrx = SensingMatrix.construct(type, args);
  
  if isempty(nxt)
    return;
  end
  
  nnclp = sens_mtrx.nNoClip();
  nxt.args.n_cols = args.n_rows - nnclp;
  nxt.args.dims = args.dims;
  nxt.args.rnd_seed = args.rnd_seed + 1;
  if ~isfield(nxt, 'nxt')
    nxt.nxt = [];
  end
  
  nxt_mtrx = genMsrmntMatrix(nxt.type, nxt.args, nxt.csr, nxt.nxt);
  
  if nnclp > 0
    nnclp_mtx = SensingMatrixScaler(nnclp, nxt_mtrx.norm());
    nnclp_mtx.setIndcsNoClip(1:nnclp);
    nxt_mtrx = SensingMatrixBlkDiag.construct({nnclp_mtx, nxt_mtrx});
  end
  
  sens_mtrx = SensingMatrixCascade.construct(...
    {nxt_mtrx, sens_mtrx.cmpSortNoClipMtrx(), sens_mtrx});
end

