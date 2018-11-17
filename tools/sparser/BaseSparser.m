classdef BaseSparser < CompMode
  %   BaseSparser - Base class for sparser classes
  %
  %   A sparsing transform is a sparsifying linear transformation T which
  %   maps the signal vector of compressive sensing into a space in which the
  %   signal is sparse (the space may be of a different dimension from the
  %   original vector).  A Sparser object contains functions to perform this
  %   transform as well as some related operations, such as:
  %   *  Multiplying by the transpose of the sparsifying matrix
  %   *  Solving the minimization problem and computing the error, the
  %      Lagrangian and other paramerters
  %
  %   The minimization problem may be defined as follows.  Let v be a
  %   reference sparse vector, e.g. one derived from a signal vector x by
  %   applying the sparsifying matrix and let s be a vector of Lagrange
  %   multipliers and let beta>0 be a penalty coefficient.  The Lagrangian
  %   is defined by;
  %
  %      l(w)=J(w) + s'*(w-v) + (beta/2)((w-v)'(w-v))
  %
  %   Where J(w) is some target function.  The object is to calculate a
  %   sparse vector w which minimizes l(w).
  %
  %   BaseSparser is a nearly trivial Sparser class which is intended to be
  %   used as a base class for more elaborate classes.  In this sparser the
  %   sparsing transformation is the unit matrix and phi(w) is the L1 norm of
  %   w, sum(|w_i|), thus the minimization probelm is fully separable, i.e.
  %   it can be solved separately for each component of w.
  
  properties
    blk_stt    % The state of the signal processed by the sparsifier
    vid_region;  % Defines the structure of the video region
    
    % A matrix for expanding the signal prior to applying the sparser
    expndr = [];
    wgts = [];
    n_sigvec   % dimension of the signal vector
    
    sprsr_mtx;
  end
  
  properties (SetAccess=protected, GetAccess=public)
    n_sprsvec;   % dimension of the sparse vector
  end    

  properties (Access=protected)
    % Here are the properties related to optimal shift.
    
    motn_data;  % A BlkMotnData object specifying the range of motion search.
    motn_crct;  % Start correcting for motion only after this number of searches.
                % default = 0;
    
    
    blk_sz; % Block size

    % All matrices are computed for one color block. Then they are
    % Kronecker multiplied from the left by cblk_mpx to generate the
    % needed matrix for the whole pixel vector (all color components and
    % all blocks).
    cblk_mpx = [];
    
    % The basic sparsifying matrix
    base_mtx; 
    
    % The sparsifying matrix with a shift
    sprsr0_mtx;
    
    % A matrix which spatially extends to the sides
    sptl_ext_mtx;
    
    % A matrix which moves the first 1:T-1 frames into frames 2:T, puts
    % zero in frame 1, and exends spatially.
    ref_mtx;
    
    % A matrix which keeps all frames but the first one.
    t2_mtx; 
    
    v_shft_mtx; % vertical shift matrices
    h_shft_mtx; % horizontal shift matrices
    v_ext_shft_mtx; % vertical shift matrices with extension
    h_ext_shft_mtx; % horizontal shift matrices with extension
    
    opt_offset;
    orig;
    
    lvl;
    lvl_cnt;
    lvl_wait;
    srch_cnt; % number of searches done.
    
  end
  
  methods
    function obj = BaseSparser(args)
      % Constructor
      %   args - a struct with fields necessary to create the sparsifier.
      %          at minimum this should be:
      %            vdrg - The VidRegion object
      %            expand_level - expansion level of the input to the
      %                           sparsifier
      %            expander - the expander used to multiply the input
      %                       vector before sparsifying
      %            sprs_mtx - (optional) the sparsifying matrix. Should
      %                       have same number of columns as the rows of
      %                       expander. Default: Unit matrix.
      if nargin > 0
        obj.set(args);
      end
    end
    
    % Returns a struct args which is suitable for building the same object
    function args = getArgs(obj) 
      args = struct(...
        'vdrg', obj.vid_region,...
        'b_stt', obj.blk_stt,...
        'expndr', obj.expndr,...
        'motn_spec', obj.motn_data.getArgs(),...
        'motn_crct', obj.motn_crct,...
        'sprs_mtx', obj.base_mtx);
    end
    
    function set(obj, args)
      % Perform the actual operations of the constructor. Has the same
      % arguments.
      %   args - a struct with fields necessary to create the sparsifier.
      %          at minimum this should be:
      %            vdrg - The VidRegion object
      %            expand_level - expansion level of the input to the
      %                           sparsifier
      %            expander - the expander used to multiply the input
      %                       vector before sparsifying
      %            sprs_mtx - (optional) the sparsifying matrix. Should
      %                       have same number of columns as the rows of
      %                       expander. Default: Unit matrix.
      obj.setBasicArgs(args);
      obj.expndr.use_gpu = obj.use_gpu;
      obj.expndr.use_single = obj.use_single;
      
      if isfield(args, 'sprs_mtx')
        obj.setBaseMtx(args.sprs_mtx)
      else
        obj.setBaseMtx(args);
      end
      obj.base_mtx.use_gpu = obj.use_gpu;
      obj.base_mtx.use_single = obj.use_single;
      
      obj.n_sprsvec = obj.base_mtx.nRows();
      obj.base_mtx.norm();
      obj.expndr.norm();
      obj.sprsr0_mtx = SensingMatrixCascade.constructCascade({...
        obj.base_mtx, obj.expndr});
      obj.sprsr0_mtx.use_gpu = obj.use_gpu;
      obj.sprsr0_mtx.use_single = obj.use_single;
      
      obj.norm();
      obj.sprsr_mtx = obj.sprsr0_mtx();
      obj.setShiftMtrcs(args);
    end
    
    % Returns the dimension of the sparse vector -
    function n = dimSprsVec(obj)
      n = obj.base_mtx.toCPUFloat(obj.n_sprsvec);
    end
    
    function setWgts(obj, wts)
      obj.wgts = wts;
    end
    
    % Apply the sparsifying transform
    %   Input
    %     first arg - (unused) this object
    %     sgnl      - The input signal (of dimension n_sigvec)
    %  Output
    %     sprs_vec  - sparse signal (of dimension dimSprsVec())
    function sprs_vec = compSprsVec(obj, sgnl)
      sprs_vec = obj.sprsr_mtx.multVec(sgnl);
    end
    
    % Apply the transpose of the sparsifying transform
    %   Input
    %     first arg - (unused) this object
    %     sprs_vec  - sparse signal (of dimension dimSprsVec())
    %  Output
    %     sgnl      - The input signal (of dimension n_sigvec)
    function sgnl = compSprsVecTrnsp(obj, sprs_vec)
      sgnl = obj.sprsr_mtx.multTrnspVec(sprs_vec);
    end
    
    function reset(obj)
      obj.lvl = 1;
      obj.lvl_cnt = 0;
      obj.lvl_wait = 0;
      obj.opt_offset = SimpleFractions([0,0]);
      obj.orig = SimpleFractions([0,0]);
      obj.sprsr_mtx = obj.sprsr0_mtx;
      obj.srch_cnt = 0;
    end
    
    function y = norm(obj)
      y = obj.sprsr0_mtx.norm();
    end

    function setNorm(obj, val, is_exact)
      obj.sprsr0_mtx.setNorm(val, is_exact);
    end
    
    function val = getExactNorm(obj)
      val = obj.sprsr0_mtx.getExactNorm();
    end
        
    % Compute sparse vector w (of dimension dimSprsVec()) which minimizes J(w),
    % given a reference sparser vector, Lagrange multiplier and a penalty
    % coefficient.  Specifically it finds w which minimizes:
    %    ||w||_1 + mltplr'(v - w) + (beta/2) ||v - w||_2^2
    % where v is the reference and beta is the penalty.
    % The algorithm works on each component separately and can
    % be verified by taking the derivative of J and checking separately for
    % cases of w positive, negative or zero.  Note that the algorithm is
    % independent on the way in which the constraint error where derived, hence
    % it is independent on the sparsing transform.  It is valid as long as the
    % target function J() is L1 norm.
    % Input:
    %    ~ - this object(may be used is subclasses)
    %    xvec - The vector to be sparsified
    %    mltplr - Lagrange multipliers
    %    beta - penalty coefficient
    function [sprs_vec_err, sprs_vec_norm, sprs_vec] = ...
        optimize(obj, xvec, mltplr, beta)
      ref_sprs_vec = obj.compSprsVec(xvec);
      if ~isempty(obj.wgts)
        beta_w = beta * obj.wgts;
        beta_inv = ones(size(beta_w)) ./ (beta_w .* obj.wgts);
        v = ref_sprs_vec + mltplr ./ beta_w;
      else
        beta_inv = 1./beta;
        v = ref_sprs_vec + mltplr*beta_inv;
      end
      sprs_vec = max((abs(v)-beta_inv),0).* sign(v);
      sprs_vec_err = ref_sprs_vec - sprs_vec;
      sprs_vec_norm = norm(sprs_vec,1);
    end
    
    function ret_val = optimizeShift(obj, sgnl)
      % Output:
      %   ret_val indicates the result of shift optimization:
      %      1 - shift changed
      %      0 - shift did not change, either waiting or same result
      %      -1 - finished all shift calculations
      if obj.lvl > obj.motn_data.nStages()
        ret_val = -1;
        return
      end
      
      while obj.lvl <= obj.motn_data.nStages()
        if ...
            (obj.lvl_cnt==0 && obj.lvl_wait<obj.motn_data.wait_first(obj.lvl)) ||...
            (obj.lvl_cnt>0 && obj.lvl_wait<obj.motn_data.wait_next(obj.lvl))
          
          obj.lvl_wait = obj.lvl_wait+1;
          break
        end
        
        obj.srch_cnt = obj.srch_cnt+1;
        
        ref_sgnl = obj.ref_mtx.multVec(sgnl);
        tst_sgnl = obj.t2_mtx.multVec(sgnl);
        
        opt_nrm = inf;
        offsets = obj.motn_data.compOffsets(obj.lvl, obj.opt_offset);
        offsets = offsets(:,1:2);
        for k=1:size(offsets,1)
          shft_mtrx = obj.getShiftMatrix(offsets(k,:),false);
          diff_sgnl = shft_mtrx.multVec(tst_sgnl) - ref_sgnl;
          nrm = norm(diff_sgnl, 1);
          if nrm < opt_nrm
            opt_nrm = nrm;
            opt_ofst = offsets(k,:);
          end
        end
        
        obj.opt_offset = normalize(opt_ofst);
        fprintf('Sparser: lvl=%d cnt=%d offset=%s\n', ...
          obj.lvl, obj.lvl_cnt, show_str(double(obj.opt_offset)));
        
        obj.lvl_wait = 0;
        obj.lvl_cnt = obj.lvl_cnt + 1;
        if obj.lvl_cnt == obj.motn_data.srch_cnt(obj.lvl)
          obj.lvl = obj.lvl+1;
          obj.lvl_cnt = 0;
        end
        
      end
      
      if obj.motn_crct && obj.srch_cnt >= obj.motn_crct
        if all(double(obj.opt_offset) == [0,0])
          shft_mtx = SensingMatrixUnit(obj.n_sigvec);
        else
          shft_mtx = getLinearShiftMatrix(obj ,-obj.opt_offset);
        end
        nrm = obj.sprsr_mtx.norm();
        obj.sprsr_mtx = SensingMatrixCascade.constructCascade({
          obj.base_mtx, shft_mtx, obj.expndr});
        obj.sprsr_mtx.setNorm(nrm);
        ret_val = 1;
      else
        ret_val = 0;
      end
      
      if ret_val == 1
        if all(obj.opt_offset == obj.orig)
          ret_val = 0;
        else
          obj.orig = obj.opt_offset;
        end
      end
    end
    
  end
  
  methods (Access=protected)
    function otr = copyElement(obj)
      otr = copyElement@CompMode(obj);
      
      otr.expndr = obj.expndr.copy();
      if isempty(obj.cblk_mpx)
        otr.cblk_mpx = obj.cblk_mpx;
      else
        otr.cblk_mpx = obj.cblk_mpx.copy();
      end
      otr.base_mtx = obj.base_mtx.copy();
      otr.sprsr0_mtx = obj.sprsr0_mtx.copy();
      if isempty(obj.sptl_ext_mtx)
        otr.sptl_ext_mtx = obj.sptl_ext_mtx;
      else
        otr.sptl_ext_mtx = obj.sptl_ext_mtx.copy();
      end
      if isempty(obj.ref_mtx)
        otr.ref_mtx = obj.ref_mtx;
      else
        otr.ref_mtx = obj.ref_mtx.copy();
      end
      if isempty(obj.t2_mtx)
        otr.t2_mtx = obj.t2_mtx;
      else
        otr.t2_mtx = obj.t2_mtx.copy();
      end
      otr.v_shft_mtx = duplicate(obj.v_shft_mtx);
      otr.h_shft_mtx = duplicate(obj.h_shft_mtx);
      otr.v_ext_shft_mtx = duplicate(obj.v_ext_shft_mtx);
      otr.h_ext_shft_mtx = duplicate(obj.h_ext_shft_mtx);
    end
    
    function setBasicArgs(obj, args)
      obj.vid_region = args.vdrg;
      obj.blk_stt = args.b_stt;
      obj.expndr = args.expndr;
      obj.n_sigvec = obj.expndr.nCols();
      if isfield(args, 'use_gpu')
        obj.use_gpu = args.use_gpu;
      end
      if isfield(args, 'use_single')
        obj.use_single = args.use_single;
      end
    end
   
    function setBaseMtx(obj, sprs_mtx)
      if nargin == 1
        obj.base_mtx = SensingMatrixUnit(obj.expndr.nrows());
      elseif sprs_mtx.nCols() ~= obj.expndr.nRows()
        error('sparser base Mtrx nCols = %d not equal tp expander mRpws = %d',...
          sprs_mtx.nCols(), obj.expndr.nRows());
      else
        obj.base_mtx = sprs_mtx.copy();
      end        
    end
    
    
    function setShiftMtrcs(obj, args)
      % This function is called during construction in order to construct
      % the matrices base_mtx, cblk_mpx, sptl_ext_mtx, ref_mtx, and the
      % matricess arrays v_shft_mtx, h_shft_mtx.
      if ~isfield(args, 'motn_spec')
        obj.motn_data = BlkMotnData(struct(...
          'm_range',[], 'm_step_numer',[], 'm_step_denom',[],...
          'm_srch_cnt',[], 'm_wait_first', [], 'm_wait_next', []));
      elseif isa(args.motn_spec, 'BlkMotnData')
        obj.motn_data = args.motn_spec.copy();
      else
        obj.motn_data = BlkMotnData(args.motn_spec);
      end
      
      if ~isfield(args, 'motn_crct')
        obj.motn_crct = 0;
      else
        obj.motn_crct = args.motn_crct;
      end
      
      if ~obj.motn_data.nStages()
        return
      end
      
      prms = obj.vid_region.getParams_vecToBlks(obj.blk_stt);
      if ~VidRegion.sameBlk(prms)
        error('not all blocks are the same');
      end
      
      % All matrices are computed for one color block. Then they are
      % Kronecker multiplied from the left by cblk_mpx to generate the
      % needed matrix for the whole pixel vector (all color components and
      % all blocks).
      obj.cblk_mpx = SensingMatrixUnit(numel(prms.ofsts));
      obj.cblk_mpx.use_gpu = obj.use_gpu;
      obj.cbl_mpx.use_single = obj.use_single;
      
      % block size
      sz = reshape(prms.blk_size(1,:,1),1,3);
      obj.blk_sz = sz;
      rg = obj.motn_data.m_range(1,:);
      extrp = cell(1,2);
      excs = cell(1,4);
      for k=1:2
        extnd_mtx = SensingMatrixDC.constructDC(rg(k),true);
        extnd_mtx.setNoClipFlag(false);
        slct_b_mtx = SensingMatrixSelect(1,sz(k));
        slct_e_mtx = SensingMatrixSelect(sz(k),sz(k));
        extrp{k} = SensingMatrixConcat({...
          SensingMatrixCascade.constructCascade({extnd_mtx, slct_b_mtx}),...
          SensingMatrixUnit(sz(k)),...
          SensingMatrixCascade.constructCascade({extnd_mtx, slct_e_mtx})});
        excs{k} = SensingMatrixSelectRange(rg(k)+1,sz(k)-rg(k), sz(k));
      end
      excs{3} = SensingMatrixSelectRange(1,sz(3)-1,sz(3));
      excs{4} = obj.cblk_mpx;
      
      obj.sptl_ext_mtx = SensingMatrixKron.constructKron(...
        [{obj.cblk_mpx, SensingMatrixUnit(sz(3))} extrp(2:-1:1)]);
      obj.ref_mtx = SensingMatrixKron.constructKron(excs(4:-1:1));
      obj.t2_mtx = SensingMatrixKron.constructKron({obj.cblk_mpx,...
        SensingMatrixSelectRange(2,sz(3),sz(3)),...
        SensingMatrixUnit(sz(2)), SensingMatrixUnit(sz(1))});
           
      % set up matrices for 1 dimensional integer shift
      [obj.v_shft_mtx, obj.v_ext_shft_mtx] = get1dShftMtxs(1);
      [obj.h_shft_mtx, obj.h_ext_shft_mtx] = get1dShftMtxs(2);
      
      obj.opt_offset = SimpleFractions([0,0]);
      
      function [shft_mtx, ext_shft_mtx] = get1dShftMtxs(d)
        % Create a cell array of one dimensional shift matrices
        shft_mtx = cell(1, 2*obj.motn_data.m_range(1,d)+1);
        ext_shft_mtx = shft_mtx;
        indx = 0;
        for s = -obj.motn_data.m_range(1,d):obj.motn_data.m_range(1,d)
          indx = indx+1;
          shft_mtx{indx} = SensingMatrixSelectRange(...
          rg(d)+s+1, sz(d)-rg(d)+s, sz(d));
          ext_shft_mtx{indx} = SensingMatrixSelectRange(...
          rg(d)+s+1, sz(d)+rg(d)+s, sz(d)+2*rg(d));
        end
      end
      
    end
    
    function shft_mtx = getShiftMatrix(obj,shft, is_ext)
      shft_mtx = obj.get2DShiftMatrix(shft, is_ext);
      shft_mtx = SensingMatrixKron.constructKron({
        obj.cblk_mpx, SensingMatrixUnit(obj.blk_sz(3)-1), shft_mtx});
    end
    
    % Get the 2D shift matrix that corresponds to the shift shft.
    % shft is a SimpleFractions object of dimension [1,2], where
    % components are [vertical, horizontal]. The is_ext flag indicates
    % whether we use the extended or not extended shift matrices
    function shft_mtx = get2DShiftMatrix(obj,shft, is_ext)
      if is_ext
        v_mtx = get1Dmtx(1, obj.v_ext_shft_mtx);
        h_mtx = get1Dmtx(2, obj.h_ext_shft_mtx);
      else
        v_mtx = get1Dmtx(1, obj.v_shft_mtx);
        h_mtx = get1Dmtx(2, obj.h_shft_mtx);
      end
      
      shft_mtx = SensingMatrixKron.constructKron({h_mtx, v_mtx});
      
      function mtx = get1Dmtx(d, shft_mtxs)
        rg = obj.motn_data.m_range(1,d);
        s = shft(d) + rg + 1;
        if s.isInt()
          s = double(s);
          mtx = shft_mtxs{s};
        else
          s0 = floor(s);
          f = double(s) - s0;
          mtx0 = shft_mtxs{s0};
          mtx1 = shft_mtxs{s0+1};
          mtx = SensingMatrixCombine([1-f;f], {mtx0, mtx1});
        end
      end
    end
    
    function shft_mtx = getLinearShiftMatrix(obj,shft)
      mtrcs = cell(obj.blk_sz(3),1);
      for k=1:length(mtrcs)
        mtrcs{k} = obj.get2DShiftMatrix(shft*(k-1), true);
      end
      shft_mtx = SensingMatrixCascade.constructCascade({...
        SensingMatrixKron.constructKron({...
        obj.cblk_mpx, SensingMatrixBlkDiag(mtrcs)}), obj.sptl_ext_mtx});
    end
    
    function setUseGpu(obj, val)
      obj.expndr.use_gpu = val;
      obj.base_mtx.use_gpu = val;
      obj.sprsr0_mtx.use_gpu = val;
      obj.sprsr_mtx.use_gpu = val;
      obj.vid_region.blkr.use_gpu = val;
      
      if ~isempty(obj.cblk_mpx)
        obj.cblk_mpx.use_gpu = val;
      end
      if ~isempty(obj.sptl_ext_mtx)
        obj.sptl_ext_mtx.use_gpu = val;
      end
      if ~isempty(obj.ref_mtx)
        obj.ref_mtx.use_gpu = val;
      end
      if ~isempty(obj.t2_mtx)
        obj.t2_mtx.use_gpu = val;
      end
      if ~isempty(obj.v_shft_mtx)
        obj.v_shft_mtx.use_gpu = val;
      end
      if ~isempty(obj.h_shft_mtx)
        obj.h_shft_mtx.use_gpu = val;
      end
      if ~isempty(obj.v_ext_shft_mtx)
        obj.v_ext_shft_mtx.use_gpu = val;
      end
      if ~isempty(obj.h_ext_shft_mtx)
        obj.h_ext_shft_mtx.use_gpu = val;
      end
    end
     
    function setUseSingle(obj, val)
      obj.expndr.use_single = val;
      obj.base_mtx.use_single = val;
      obj.sprsr0_mtx.use_single = val;
      obj.sprsr_mtx.use_single = val;
      obj.vid_region.blkr.use_single = val;
      
      if ~isempty(obj.cblk_mpx)
        obj.cblk_mpx.use_single = val;
      end
      if ~isempty(obj.sptl_ext_mtx)
        obj.sptl_ext_mtx.use_single = val;
      end
      if ~isempty(obj.ref_mtx)
        obj.ref_mtx.use_single = val;
      end
      if ~isempty(obj.t2_mtx)
        obj.t2_mtx.use_single = val;
      end
      if ~isempty(obj.v_shft_mtx)
        obj.v_shft_mtx.use_single = val;
      end
      if ~isempty(obj.h_shft_mtx)
        obj.h_shft_mtx.use_single = val;
      end
      if ~isempty(obj.v_ext_shft_mtx)
        obj.v_ext_shft_mtx.use_single = val;
      end
      if ~isempty(obj.h_ext_shft_mtx)
        obj.h_ext_shft_mtx.use_single = val;
      end
      
      obj.lvl = uint32(obj.lvl);
      obj.lvl_cnt = uint32(obj.lvl_cnt);
      obj.lvl_wait = uint32(obj.lvl_wait);
      obj.srch_cnt = uint32(obj.srch_cnt);
    end
  end
  
  methods (Static)
    % Create a new sparsifier
    %   Input:
    %     name - class of the sparsifier (char)
    %     args - a struct with fields necessary to create the sparsifier.
    %            at minimum this should be:
    %              vdrg - The VidRegion object
    %              expand_level - expansion level of the input to the
    %                             sparsifier
    %              expander - the expander used to multiply the input
    %                         vector before sparsifying
    %              unexpander - (optional) inverse of the expander. If
    %                           present, normalization is done.
    %   Output:
    %     sprsr - the sparsifier.
    function sprsr = construct(name, args)
      sprsr = eval(name);
      sprsr.set(args);
    end
    
  end
  
end

