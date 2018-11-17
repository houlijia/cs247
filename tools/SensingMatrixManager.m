classdef SensingMatrixManager < handle
  properties (Access=protected)
    % CS_EncParams object
    enc_opts;
    
    % VidBlocker
    vid_blocker;
    
    % Repositories for sensing matrices
    blk_sens_mtrx_repos = KeyedRepository; % block specific
    frm_sens_mtrx_repos = KeyedRepository; % whole frame

  end
  
  properties (SetAccess = protected)
    blk_cnt
    
    % No. of colomns and rows in each block matrix, based on its position
    ncols;
    nrows;
    msrs_bgn;
    msrs_end;
  end
  
  methods
    function obj = SensingMatrixManager(e_opts, blkr)
      % Constructor
      %   Input:
      %     e_opts - an CS_EncParams object
      %     blkr - A VidBlocker object
      
      obj.enc_opts = e_opts;
      obj.vid_blocker = blkr;
      obj.blk_cnt = blkr.blk_cnt;
      
      obj.ncols = zeros(obj.blk_cnt(1), obj.blk_cnt(2));
      
      zero_ext = struct('b',obj.enc_opts.zero_ext_b, ...
      'f', obj.enc_opts.zero_ext_f, 'w', obj.enc_opts.wrap_ext);
      
      for h=1:obj.blk_cnt(2)
        for v=1:obj.blk_cnt(1)
          vid_region = VidRegion([v,h,1], blkr, zero_ext);
          obj.ncols(v,h) = vid_region.ext_vec_len;
        end
      end
      obj.nrows = ceil(obj.vid_blocker.cntOrigRawPxls * e_opts.msrmnt_input_ratio);
      obj.nrows = reshape(obj.nrows, size(obj.ncols));
      
      if e_opts.par_blks > 0
        obj.msrs_end = obj.nrows(:) * ones(1,e_opts.par_blks);
        obj.msrs_end = cumsum(obj.msrs_end(:));
        obj.msrs_bgn = [1; (obj.msrs_end(1:end-1)+1)];
      end

      % Initialize matrices and repositories
      nt = max(e_opts.random.rpt_temporal, max(1, e_opts.par_blks));
      if e_opts.random.rpt_spatial
        ns = 9;
      else
        ns = obj.blk_cnt(1)* obj.blk_cnt(2);
      end
      obj.blk_sens_mtrx_repos.init(nt*ns, 3);
            
      if e_opts.random.rpt_temporal
        % Fill the the repository
        for t = 1:min(e_opts.random.rpt_temporal, obj.blk_cnt(3))
          for h=1:obj.blk_cnt(2)
            for v=1:obj.blk_cnt(1)
              obj.getBlockMtrx([v,h,t]);
            end
          end
        end
        
        if obj.enc_opts.par_blks > 0
          obj.frm_sens_mtrx_repos.init(e_opts.random.rpt_temporal, 2);
          for r=1:min(e_opts.random.rpt_temporal, ...
              floor(obj.blk_cnt(3)/obj.enc_opts.par_blks))
            
            t_ofst = (r-1)*obj.enc_opts.par_blks;
            obj.getFrmsMtrx(t_ofst+1, t_ofst+obj.enc_opts.par_blks);
          end
        end
      end
    end
    
    function sens_mtrx = getBlockMtrx(obj, blk_indx, vid_region)
      seed = obj.enc_opts.randomSeed(blk_indx, obj.blk_cnt);
      n_cols = obj.ncols(blk_indx(1), blk_indx(2));
      n_rows = obj.nrows(blk_indx(1), blk_indx(2));
      mt_key = [seed.seed, n_cols, n_rows];
      sens_mtrx = obj.blk_sens_mtrx_repos.get(mt_key);
      if isempty(sens_mtrx)
        if nargin < 3
          zero_ext = struct('b',obj.enc_opts.zero_ext_b, ...
            'f', obj.enc_opts.zero_ext_f, 'w', obj.enc_opts.wrap_ext);
          vid_region = VidRegion(blk_indx, obj.vid_blocker, zero_ext);
        end
      
        sens_mtrx = obj.enc_opts.constructSensingMatrix(vid_region, n_rows);
        sens_mtrx.use_gpu = obj.enc_opts.use_gpu;
        sens_mtrx.use_single = obj.enc_opts.use_single;
        obj.blk_sens_mtrx_repos.put(sens_mtrx, mt_key);
      end
    end
    
    function sens_mtrx = getFrmsMtrx(obj, frst, last)
      sens_mtrx = [];
      n_frms = last-frst+1;
      if obj.enc_opts.random.rpt_temporal
        frm_mt_key = [mod(frst, obj.enc_opts.random.rpt_temporal), n_frms];
        sens_mtrx = obj.frm_sens_mtrx_repos.get(frm_mt_key);
      else
        frm_mt_key = [];
      end
      if isempty(sens_mtrx)
        mtrcs = cell(obj.blk_cnt(1), obj.blk_cnt(2), n_frms);
        for t=1:n_frms
          for h=1:obj.blk_cnt(2)
            for v=1:obj.blk_cnt(1)
              mtrcs{v,h,t} = obj.getBlockMtrx([v,h,t+frst-1]);
            end
          end
        end
        sens_mtrx = SensingMatrixBlkDiag.construct(mtrcs(:));
        if ~isempty(frm_mt_key)
          obj.frm_sens_mtrx_repos.put(sens_mtrx, frm_mt_key);
        end
      end
    end
  end
end