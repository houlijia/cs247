classdef VidRegion < CodeElement
  % VidRegion is a description of video region.
  % It contains a list of video block indices and information about the block
  % sizes. However, the object does not contain the pixel values or
  % equivalent information.
  %
  % When methods of VidRegion deal with pixels it is assumed that they are
  % organized as an array of cells indexed by (iclr,iblk) where iclr is
  % the color component (YUV) and iblk is the block number corresponding
  % to blk_indx, that is obj.blk_indx(:,iblk) is the position of the
  % block in the video volume. Each cell contains a 3D pixel array of size
  % obj.clrBlkSize()
  %
  % VidRegion provides the function vectorize() to convert a list of blocks
  % into a vector and pixelize() to convert a vector into a sequence of blocks.
  
  
  properties
    n_color=1;     % Can be 1 or 3 for color or BW.
    n_blk = 0;  % No. of blocks in the region.
    
    % array of n_blk indices of size [n_blk,3].  Each row contains the
    % [vertical, horizontal, temporal] index of a block.
    blk_indx = [];
    
    % zero extension of a block.  It is a 2x3 array, where
    % the first row is the backward extnesion and second is
    % the forward extension.  Columns represent (H,V,T).
    zext = [0 0 0; 0 0 0];
    
    % Wrap extension of a block
    wext = [0 0 0];
    
    blkr           % A VidBlocker object
    vec_len=0;     % Length of vector created from pixels
    ext_vec_len=0; % Length of extended vector created from pixels
    % (including zeroes)
    
    % The following determine sizes based on the state. Here state is
    % one of BLK_STT_xxx, color is 1 - n_color and dim is one of
    % verical, horizontal, temporal.
    stt_Cblk_size=[]; % Color block size, indexed by (color,dim,blk_no,state)
    stt_Cblk_len=[];  % Pixels in color block, indexed by (color, blk_no, state)
    stt_blk_len=[];  % Total pixels in block, indexed by (blk_no, state);
    stt_vec_len=[];  % No. of pixels in the region, indexed by (state).
    stt_Cblk_offset=[]; % Offset to the beginning of a color block in the
    % pixel vector, indexed by (color, blk_no,state)
  end
  
  properties (SetAccess=private)
    % These properties get their value by the get function, on the first
    % call
    
    % Number of pixels (before blocking)
    n_clr_pxls=[];
    n_pxls=[];
    
    % Number of original pixels (may be a fraction because of overlap) in
    % each block and color. Indexed by (color, blk)
    n_orig_blk_clr_pxls=[];
    
    % Number of original pixels (may be a fraction because of overlap) in
    % each block. Indexed by (blk)
    n_orig_blk_pxls=[];
    
    % True if the region consists of consecutive whole frames, with block in an
    % increasing order
    whole_frames;
  end
  
  properties (Access=private)
    % These properties get their value by the get function, on the first
    % call
    
    % A cell array (one cell for each color) of 3D arrays corresponding to
    % pixels. Each entry contains the number of blocks in which each pixel
    % is included.
    pxl_cnt_map = [];
    
    % Arrays indexed by (color, dimension, blk_no) indicating the beginning
    % and end of each 
    
    % Offset of pxl_cnt_map from beginning in the video (n_color,3).
    ofst_pxl_cnt_map = [];
    
    % Matrix to expand from BLK_STT_NONE to BLK_STT_RAW
    expnd_pxl_mat = [];
    
    % Matrix to expand from BLK_STT_NONE to BLK_STT_RAW
    unexpnd_pxl_mat = [];
    
  end  
  
  methods
    % Constructor
    %   Can have 0,2,3 or 4 arguments. If there are no arguments a dummy object
    %   is created.
    %     Input arguments:
    %       blk_indices - value for blk_indx
    %       vid_blocker - value for obj.blkr
    %       zero_ext - optional argument for zext
    %               [zero_ext_b; zero_ext_f].  If it has 3 rows the
    %               the third row is wrap_ext. It can also be a struct
    %               with fields 'b','f','w' specifying zero_ext_b,
    %               zero_ext_f and wrap_ext.
    %       wrap_ext - optional argument for wext (cannot be present if
    %               wrap_ext was specified in zero_ext).
    function obj = VidRegion(blk_indices, vid_blocker, zero_ext, wrap_ext)
      switch (nargin)
        case 0
          return
        case 2
          zero_ext = obj.zext;
          wrap_ext = obj.wext;
        case 3
          if isstruct(zero_ext)
            wrap_ext = zero_ext.w;
            zero_ext = [zero_ext.b; zero_ext.f];
          elseif size(zero_ext,1) >= 3
            wrap_ext = zero_ext(3,:);
            zero_ext = zero_ext(1:2,:);
          else
            wrap_ext = obj.wext;
          end
        case 4;  % Do nothing
        otherwise
          error('Illegal number of arguments');
      end
      obj.init(blk_indices, vid_blocker, zero_ext, wrap_ext);
    end
    
    function init(obj, blk_indices, vid_blocker, zero_ext, wrap_ext)
      obj.n_blk = size(blk_indices,1);
      obj.blk_indx = blk_indices;
      obj.blkr = vid_blocker;
      obj.n_color = size(vid_blocker.blk_size,1);
      obj.zext = zero_ext;
      obj.wext = wrap_ext;
      obj.setBlksLength();
      
      % Set whole_frames
      blk_cnt = obj.blkr.blk_cnt;
      obj.whole_frames = all(obj.blk_indx(1,1:2)==[1 1]) &&...
        all(obj.blk_indx(end,1:2) == blk_cnt(1:2));
      if obj.whole_frames
        % Check consecutiveness
        for k=1:(size(obj.blk_indx,1)-1)
          if obj.blk_indx(k,1) < blk_cnt(1)
            if any(obj.blk_indx(k+1,:) ~= obj.blk_indx(k,:) + [0,0,1])
              obj.whole_frames = false;
              break
            end
          elseif obj.blk_indx(k,2) < blk_cnt(2)
            if any(obj.blk_indx(k+1,:) ~= ...
                [1, obj.blk_indx(k,2)+1, obj.blk_indx(3)])
              obj.whole_frames = false;
              break
            end
          elseif any(obj.blk_indx(k+1,:) ~= [1,1,obj.blk_indx(k,3)+1])
              obj.whole_frames = false;
              break
          end
        end
      end
    end
    
    function vdrg = getSingleBlk(obj, indx)
      % Compute a VidRegion object for the single block of index indx.
      vdrg = VidRegion();

      vdrg.n_blk = 1;
      vdrg.blk_indx = obj.blk_indx(1,:);
      vdrg.blkr = obj.blkr;
      vdrg.n_color = obj.n_color;
      vdrg.zext = obj.zext;
      vdrg.wext = obj.zext;

      vdrg.vec_len = obj.vec_len / obj.n_blk;
      vdrg.ext_vec_len = obj.ext_vec_len / obj.n_blk;

      vdrg.stt_Cblk_size = reshape(obj.stt_Cblk_size(:,:,indx,:),...
        size(obj.stt_Cblk_size,1), size(obj.stt_Cblk_size,2),...
        1, size(obj.stt_Cblk_size,4));
      vdrg.stt_Cblk_len = reshape(obj.stt_Cblk_len(:,indx,:),...
        size(obj.stt_Cblk_len,1), 1, size(obj.stt_Cblk_len,3));
      vdrg.stt_blk_len = reshape(obj.stt_blk_len(indx,:),...
        1, size(obj.stt_blk_len,2));
      cboff = cumsum(vdrg.stt_Cblk_len,1);
      vdrg.stt_Cblk_offset = ...
        cat(1, zeros(1, 1, size(vdrg.stt_Cblk_len,3)), cboff(end-1,:,:));
      
      vdrg.whole_frames = obj.whole_frames && all(obj.blkr.blk_cnt(1:2) == 1);
    end
    
    function empty_vid = getEmpty(obj)
      empty_vid = cell(obj.n_color, obj. n_blk);
      for iclr = 1:obj.n_color
        for iblk = 1:obj.n_blk
          empty_vid{iclr,iblk} = zeros(obj.blkr.blk_size(1,:));
        end
      end
    end
    
    % Get the pixels of a video region from a blocks cell-array
    % Input:
    %   obj - this VideoRegion
    %   blk_array - A 4-dim cell array of blocks (color, vertical,
    %   horizontal, temporal)
    %   blk_start - (optional) offset block numbers in blk_array.
    %               Default = [0,0,0].
    % Output
    %   The cell array of the region.
    function out = getFromBlkArray(obj, blk_array, blk_start)
      if nargin < 3
        blk_start = [0,0,0];
      end
      
      out = cell(obj.n_color, obj.n_blk);
      %             avail_colors = min(obj.n_color, size(blk_array,1));
      for i_blk = 1:obj.n_blk
        blk_ind = obj.blk_indx(i_blk,:) - blk_start;
        v_ind = blk_ind(1);
        h_ind = blk_ind(2);
        t_ind = blk_ind(3);
        for i_clr = 1:obj.n_color %avail_colors
          out{i_clr, iblk} = double(blk_array{i_clr, v_ind, h_ind,...
            t_ind});
        end
      end
    end
    
    % Put the pixels of a video region into a VidBlocksOut object
    % Input:
    %   obj - this VideoRegion
    %   vec - the pixels to put in (a vector or a cell array
    %   (blk_no,color).
    %   blk_array - a VidBlocksOut object.
    % Output
    %   blk_array - The updated block array.
    function blk_array = putIntoBlkArray(obj, vec, blk_array)
      if ~iscell(vec)
        vec = obj.pixelize(vec, blk_array.blk_stt);
      end
      
      for i_blk = 1:obj.n_blk
        blk_array.insertBlk(vec(:,i_blk), obj.blk_indx(i_blk,:));
      end
    end
    
    % put a value into a 3-array, where each entry represents a blocks,
    % for each block in this region
    % Input:
    %   obj - this VideoRegion
    %   blk_map - the 3D array 
    %   val - the value to insert into the array
    %   ofst - temporal (3rd dim) offset
    % Output
    %   blk_map - The updated block array.
    function blk_map = markBlkMap(obj, blk_map, val, ofst)
      for i_blk = 1:obj.n_blk
        blk_ind = obj.blk_indx(i_blk,:);
        v_ind = blk_ind(1);
        h_ind = blk_ind(2);
        t_ind = blk_ind(3) - ofst;
        blk_map(v_ind, h_ind, t_ind) = val;
      end
    end
    
    % Return a single block size
    function bsize = clrBlkSize(obj)
      bsize = obj.blkr.blk_size(1,:);
    end
    
    function dims = spatioTemporalDimensions(obj, stt)
      % Compute the spatio-temporal dimensions of region
      %   Input:
      %     obj: This object
      %     stt: (optional) The exspansion state of the data (one of
      %           VidBlocker.BLK_STT_xxx). Default: BLK_STT_ZEXT
      %   Output:
      %     dims: A hoizontal vector with 3 components:
      %           1 - number of entries per spatial frame (in one color)
      %           2 - Number of temporal frames in a block
      %           3 - Number of color blocks, where each color block is a
      %               one color coponent of th eblock
      if nargin < 2
        stt = VidBlocker.BLK_STT_ZEXT;
      end
      
      sz = obj.stt_Cblk_size(:,:,:,stt);
      for dm=1:size(sz,2)
        if any(any(sz(:,dm,:) ~= sz(1,dm,1)))
          error(['stt_Cblk_size(color,%d,block_no,%d) not constant '...
            'over colors and blocks'], dm, stt);
        end
      end
      
      dims = [sz(1,1,1)*sz(1,2,1), sz(1,3,1), obj.n_color*obj.n_blk];
    end
    
    % Convert a vector into a array of blocks, including removal of
    % zero extension
    % Input
    %   obj:  This object
    %   vec: A vector of length obj.vec_len
    %   stt: State of vector (BLK_STT_xxx). Default = BLK_STT_WNDW
    % Output
    %   blks: A cell array of size [obj.n_color, obj.n_blk].  Each cell
    %     contains a 3-D array of pixels organized as [Vertical, Horizontal,
    %     Temporal].  blks{i,j} contains the pixels of block j, component i,
    %     where i=1,2,3 corresponds to Y,U,V
    
    function blks = pixelize(obj, vec, stt)
      if nargin < 3
        stt = obj.blkr.BLK_STT_WNDW;
      end
      
      prms = obj.getParams_vecToBlks(stt);
      blks = obj.vecToBlks(vec, prms);
    end
    
    function prms = getParams_vecToBlks(obj, stt)
      prms = struct(...
        'ofsts', obj.stt_Cblk_offset(:,:,stt),...
        'blk_len', obj.stt_Cblk_len(:,:,stt),...
        'blk_size', obj.stt_Cblk_size(:,:,:,stt));
    end
    
    function n_pxl = nOrigPxlInRegion(obj)
      % Compute total number of real pixels (before extension and interpolation)
      % in the region
      % Input:
      %   obj - this object
      
      n_pxl = sum(obj.n_orig_blk_pxls);
    end
    
    function pos_indx = indexPosition(obj)
      pos_indx = obj.blkr.indexPosition(obj.blk_indx);
    end
   
    % Compute matrices which expand (or unexpand) a vector of video at
    % a given state into other states.
    % Input:
    %   obj - this object
    %   inp_stt - state of input (BLK_STT_x)
    %   out_stt - state of output (BLK_STT_x)
    %   eps - threshold for setting low singular values to zero
    % Output
    %   xpnd - a struct with component 'M', 'U', 'L', 'V', which are objects
    %          of type SensingMatrix. xpnd.M is the required matrix. The
    %          'U', 'L' and 'V' fields are the singular value decomposition
    %          (SVD) of xpnd.M. Thus:
    %            xpnd.M = xpnd.U * xpnd.L * xpnd.V
    %          where xpnd.L is diagonal (of type SensingMatrixDiag), the
    %          columns of xpnd.U are orthonormal and the rows of xpnd.V are
    %          orthonormal.
    xpnd = getExpandMtrx(obj, inp_stt, out_stt, eps);
    
    % Compute constraints matrix (must satisfy Cx=0)
    %   Input:
    %     obj - this object
    %     stt - state of vector(BLK_STT_x)
    %   output
    %     cnstrnts - constraints matrix
    cnstrnts = compConstraints(obj,stt)
    
    % Compute edge detection on a vector of pixels
    % Input
    %   inp - the input vector (in vector or pixel format)
    %   dim_cnt - an array of [v,h,t], defining the number of neighbors
    %             to average on each side
    %   stt - State of input (BLK_STT_xxx)
    % Output
    %   out - edge detected signal, in the same format and dimension as
    %         inp.
    function out = edgeDtct(obj, inp, dim_cnt, stt)
      cnt = prod(2*dim_cnt+1) - 1;
      if ~cnt
        out = inp;
        return
      end
      if iscell(inp)
        out = inp;
      else
        out = obj.pixelize(inp, stt);
      end
      for iclr = 1:obj.n_color
        for iblk=1:obj.n_blk
          sz0 = size(out{iclr,iblk});
          sz = [1 1 1];
          sz(1:length(sz0)) = sz0;
          esz = sz + 2*dim_cnt;
          vb = zeros(sz);
          ob = ob{iclr,iblk};
          
          eb = zeros(esz);
          eb(1+dim_cnt(1):end-dim_cnt(1),...
            1+dim_cnt(2):end-dim_cnt(2),...
            1+dim_cnt(3):end-dim_cnt(3)) = ob;
          
          %extend
          for j=1:dim_cnt(1)
            eb(j,...
              1+dim_cnt(2):end-dim_cnt(2),...
              1+dim_cnt(3):end-dim_cnt(3)) = ...
              eb(1+dim_cnt(1),...
              1+dim_cnt(2):end-dim_cnt(2),...
              1+dim_cnt(3):end-dim_cnt(3));
            eb(end-j+1,...
              1+dim_cnt(2):end-dim_cnt(2),...
              1+dim_cnt(3):end-dim_cnt(3)) = ...
              eb(end-dim_cnt(1),...
              1+dim_cnt(2):end-dim_cnt(2),...
              1+dim_cnt(3):end-dim_cnt(3));
          end
          for j=1:dim_cnt(2)
            eb(:,j,1+dim_cnt(3):end-dim_cnt(3)) = ...
              eb(:,1+dim_cnt(2),...
              1+dim_cnt(3):end-dim_cnt(3));
            eb(:,end-j+1,1+dim_cnt(3):end-dim_cnt(3)) = ...
              eb(:,end-dim_cnt(2),1+dim_cnt(3):end-dim_cnt(3));
          end
          for j=1:dim_cnt(3)
            eb(:,:,j) = eb(:,:,1+dim_cnt(3));
            eb(:,:,end-j+1) = eb(:,:,end-dim_cnt(3));
          end
          
          % Compute sum of neighbors
          for v=-dim_cnt(1):dim_cnt(1)
            for h=-dim_cnt(2):dim_cnt(2)
              for t =-dim_cnt(3):dim_cnt(3)
                vb = vb + eb(1+dim_cnt(1)+v:end-dim_cnt(1)+v,...
                  1+dim_cnt(2)+h:end-dim_cnt(2)+h,...
                  1+dim_cnt(3)+t:end-dim_cnt(3)+t);
              end
            end
          end
          
          out{iclr,iblk} = ob - (vb-ob)/cnt;
        end
      end
      if ~iscell(inp)
        out = obj.vectorize(out);
      end
    end
    
    % Convert pixel offsets to vector index offsets
    % Input:
    %   pxl_ofst - an array of size (m,3) where each row defines one
    %   offset - (V,H,T).
    %   use_zext - If true take zero extension into account
    % Output
    %   vec_ofst a vector of size(m,1) containing the offsets
    function vec_ofst = offsetPxlToVec(obj, pxl_ofst, use_zext)
      if use_zext
        cblk_size = obj.blkr.ext_Cblk_size;
      else
        cblk_size = obj.blkr.blk_size(1,:);
      end
      vec_ofst = pxl_ofst(:,2) + pxl_ofst(:,3)*cblk_size(2);
      vec_ofst = pxl_ofst(:,1) + vec_ofst*cblk_size(1);
    end
    
    % Draw motion marker
    % insert motion indicator into all blocks of the region
    % Input:
    %   vec - a cell array or vector
    %   blk_stt - State of vector (BLK_STT_xxx)
    %   mrkr_pos - Marker position in the block [h,v], where h,v are
    %         relative vertical and horizontal positions in [0,1]
    %   clr - color of marker (in the Y component). This is an array of 2
    %         values: The first is the color of the circle/ring indicating 
    %         confidence.  The second is the color of the line segment showing
    %         direction.
    %   cnfdnc - confidence level (pair of numbers between -1 and 1)
    %   vlcty - Velocity [h,v] in units of block size fraction per frame
    %           (i.e. motion in pixels per frame divided by block size in
    %           pixels in that dimension.
    function vec = drawMotionMarker(obj, vec, blk_stt, mrkr_pos, clr,...
        blk_motion)
      if ~iscell(vec)
        vec = obj.pixelize(vec, blk_stt);
        vec_is_vector = true;
      else
        vec_is_vector = false;
      end
      
      for iblk=1:size(obj.blk_indx,1)
        vec(:,iblk) = mark_blk_motion(vec(:,iblk), ...
          mrkr_pos, clr, [blk_motion.maxval, blk_motion.midval],...
          blk_motion.vlcty);
      end
      
      if vec_is_vector
        vec = obj.vectorize(vec);
      end
    end
    
    % Input:
    %   vec - a cell array or vector
    %   blk_stt - State of vector (BLK_STT_xxx)
    %   mrkr_pos - Marker position in the block [h,v], where h,v are
    %         relative vertical and horizontal positions in [0,1]
    %   clr - color of marker (in the Y component).
    %   cnfdnc - confidence level (pair of numbers between -1 and 1)
    %   vlcty - Velocity [h,v] in units of block size fraction per frame
    %           (i.e. motion in pixels per frame divided by block size in
    %           pixels in that dimension.
    function vec = drawStatMotionMarker(obj, vec, blk_stt, mrkr_pos, ...
        clr, activity_stat, thrhld)
      if ~iscell(vec)
        vec = obj.pixelize(vec, blk_stt);
        vec_is_vector = true;
      else
        vec_is_vector = false;
      end
      
      % for iblk=1:size(obj.blk_indx,1)
      %    vec(:,iblk) = mark_blk_motion(vec(:,iblk), ...
      %       mrkr_pos, clr, [blk_motion.maxval, blk_motion.midval],...
      %      blk_motion.vlcty);
      %end
      
      % Mark frames based on the statistics of background
      % measurements of blocks
      for iblk=1:size(obj.blk_indx,1)
        vec(:,iblk) = Stats_based_marker(vec(:,iblk), ...
          mrkr_pos, clr, activity_stat,...
          thrhld);
      end
      
      if vec_is_vector
        vec = obj.vectorize(vec);
      end
    end
    
    function vec = drawAltStatMotionMarker(obj, vec, blk_stt,...
        mrkr_pos, clr, activity_stat, thrhld)
      if ~iscell(vec)
        vec = obj.pixelize(vec, blk_stt);
        vec_is_vector = true;
      else
        vec_is_vector = false;
      end
      
      % for iblk=1:size(obj.blk_indx,1)
      %    vec(:,iblk) = mark_blk_motion(vec(:,iblk), ...
      %       mrkr_pos, clr, [blk_motion.maxval, blk_motion.midval],...
      %      blk_motion.vlcty);
      %end
      
      % Mark frames based on the statistics of background
      % measurements of blocks
      for iblk=1:size(obj.blk_indx,1)
        vec(:,iblk) = AltStats_based_marker(vec(:,iblk), ...
          mrkr_pos, clr, activity_stat,...
          thrhld);
      end
      
      if vec_is_vector
        vec = obj.vectorize(vec);
      end
    end
    
    % Compute the vector indices which can be shifted by pxl_ofst
    % Input:
    %   obj - this object
    %   pxl_ofst - an array of size (m,3) where each row defines one
    %   offset (V,H,T).
    %   blk_stt - status of vec (BLK_STT_x). Assuming >=
    %             BLK_STT_INTRPLT.
    % Output
    %   vec_indcs - a vector of indices which can shifted by all the
    %      shiftsin pxl_ofst without going  out of bound
    vec_indcs = inRngVec(obj, pxl_ofst, blk_stt)
    
    % Compute 3-D cross correlation of vector from shifted versions of the
    % vector.
    %   Input:
    %     obj - this object
    %     vec - the pixels of the region on which cross correlatios
    %           need to be computed.
    %     vec_indcs - the indices (linear) on which the norm is
    %                 computed.
    %     offsets - an array of size (m,3) where each row one offset to
    %           comptue correlation on.
    %   Output
    %     xcor - the cross correlations, a vector of size (m,1),
    %            divided (point wise) by vec0_nrm*vec_nrms;
    %     vec0_nrm - the norm of the reference vector (offset zero)
    %     vec_nrms - the norms of the shifted vectors
    [xcor, vec0_nrm, vec_nrms] = compXCor(obj, vec, vec_indcs, offsets)
    
    % Compute 3-D normed diff of vector from shifted versions of the
    % vector.
    %   Input:
    %     obj - this object
    %     vec - the pixels of the region on which cross correlatios
    %           need to be computed.
    %     vec_indcs - the indices (linear) on which the norm is
    %                 computed.
    %     offsets - an array of size (m,3) where each row one offset to
    %           comptue correlation on.
    %     nrm_exp - (optional, default=1) norm exponent.
    %   Output
    %     xcor - the cross correlations, a vector of size (m,1),
    %            divided (point wise) by vec0_nrm*vec_nrms;
    %     vec0_nrm - the norm of the reference vector (offset zero)
    [xcor, vec0_nrm] = compXDiff(obj, vec, vec_indcs, offsets, nrm_exp)
    
    % Draw a rectangle in a group of frames in each block
    % Input:
    %    obj: this object
    %    vec: The pixels of the region.  Can be either in the form of
    %         pixels array (cells) or a single vector
    %    frm_no: An array of frame numbers in which the rectangle need
    %            be drawn
    %    clr: An array of the color component indices in which the
    %         rectangles need to be drawn
    %    val: The value to set. can be a scalar or of the same length
    %         as clr.
    %    pos: position of the rectangle in the frames.  This is a 2x2 array
    %          where the first row is the beginning point of (h,v)
    %          (horizontal, vertical  or 1,2) axes and the values are
    %          relative to the frame size, i.e. between 0 and 1.
    %   lwdth: (optional)line width - a 2x2 array of line widths relative
    %            to the rectangle size. first colomn is horizontal widths
    %            and 2nd is vertical widths. The defalut width is 0, meaning
    %            one pixel
    %    keep_aspect_ratio: (optional) if true the larger dimension is reduced
    %           to keep the aspect ratio.
    % Output:
    %    vec: the modified pixels, in the same format as the input
    vec = frm_draw_rect(obj,vec,frm_no,clr,val,pos,lwdth,keep_aspect_ratio)
    
    len = encode(obj, code_dst, ~)
    len = decode(obj, code_src, info, cnt)
  end
  
  methods   % get/set functions
    function val = get.n_clr_pxls(obj)
      if isempty(obj.pxl_cnt_map)
        obj.compPxlCntMap();
      end
      val = obj.n_clr_pxls;
    end
    
    function val = get.n_pxls(obj)
      if isempty(obj.pxl_cnt_map)
        obj.compPxlCntMap();
      end
      val = obj.n_pxls;
    end
    
    function val = get.n_orig_blk_clr_pxls(obj)
      if isempty(obj.pxl_cnt_map)
        obj.compPxlCntMap();
      end
      val = obj.n_orig_blk_clr_pxls;
    end
    
    function val = get.n_orig_blk_pxls(obj)
      if isempty(obj.pxl_cnt_map)
        obj.compPxlCntMap();
      end
      val = obj.n_orig_blk_pxls;
    end
    
    function val = get.ofst_pxl_cnt_map(obj)
      if isempty(obj.pxl_cnt_map)
        obj.compPxlCntMap();
      end
      val = obj.ofst_pxl_cnt_map;
    end
    
    function val = get.expnd_pxl_mat(obj)
      if isempty(obj.expnd_pxl_mat)
        obj.compExpPxlMat();
      end
      val = obj.expnd_pxl_mat;
    end
    
    function val = get.unexpnd_pxl_mat(obj)
      if isempty(obj.unexpnd_pxl_mat)
        obj.compUnExpPxlMat();
      end
      val = obj.unexpnd_pxl_mat;
    end
    
    function out = multiDiffExtnd(obj, inp, dim_cnt, blk_stt)
      % Perform Extended differences (compDiffExntd) several times along
      % several dimensions.
      % Input:
      %   obj - this object
      %   inp - the input region. Can be a cell array of 3D arrays, or, if
      %     stt is specified, also a vector. In that case the vector is
      %     pixelized to blk_stt level before processing
      %   dim_cnt - a three dimensional array specifying how many times
      %        to compute the difference along each dimension
      %   blk_stt - Status of the input. Needed only if inp is a vector
      % Output:
      %   out - a cell array of 3D arrays.
      if ~iscell(inp) && nargin >=4
        inp = obj.pixelize(inp, blk_stt);
      end
      
      out = obj.do_multiDiffExtnd(inp, dim_cnt);
    end
        
    function out = multiDiffUnExtnd(obj, inp, dim_cnt, blk_stt)
      % Undo the operatrion of multiDiffExtnd.
      % Input:
      %   obj - this object
      %   inp - the input region. Can be a cell array of 3D arrays, or, if
      %     stt is specified, also a vector. In that case the vector is
      %     pixelized to blk_stt level before processing
      %   dim_cnt - a three dimensional array specifying how many times
      %        to compute the difference along each dimension
      %   blk_stt - Status of the input. Needed only if inp is a vector
      % Output:
      %   out - output
      if ~iscell(inp) && nargin >=4
        inp = obj.pixelize(inp, blk_stt);
      end
      
      out = obj.do_multiDiffUnExtnd(inp, dim_cnt);
    end
end
  
  methods (Static)
    % convert a blocks list into a vector, without the zero
    % extension.
    % Input
    %   blks: A cell array.  Each cell
    %     contains a 3-D array of pixels organized as [Vertical, Horizontal,
    %     Temporal].  blks{i,j} contains the pixels of block j, color 
    %     component i, where i=1,2,3 corresponds to Y,U,V
    % Output
    %   vec: A vector of the total length of the components of blks
    function vec = vectorize(blks)
      vec_blk = cell(size(blks));
      for k = 1:numel(blks)
        vec_blk{k} = blks{k}(:);
      end
      vec = vertcat(vec_blk{:});
    end
    
    % vecToBlks converts a linear vector into a cell array of 3D
    % matrices of size [n_c,n_b]
    %   Input:
    %     vec - the input vecotr. its length should be sum(blk_len(:))
    %     prms - a struct with the following fields:
    %       ofsts - an array of size [n_c, n_b], containing the offsets to
    %               each block
    %       blk_len - an array of size [n_c, n_b], containing the length of
    %                 each block
    %       blk_size - an array of size [n_c, 3, n_b], containing the 
    %                 dimensions of each block
    %  Output:
    %    blks - a cell array of size [n_b,n_c].
    function blks = vecToBlks(vec, prms)
      n_c = size(prms.ofsts,1);
      n_b = size(prms.ofsts,2);
      blks = cell(n_c, n_b);
      for iclr = 1:n_c
        for iblk = 1:n_b
          ofst = prms.ofsts(iclr,iblk);
          blks{iclr,iblk} = reshape(vec(ofst+1:ofst+prms.blk_len(iclr,iblk)), ...
            prms.blk_size(iclr,:,iblk));
        end
      end
    end
    
    % The function receives a cell array of pixels and cuts out the
    % pixels which are not represented by compDiff
    function pxls = getDiffPxls(pxls, dim, mode)
      if mode
        return
      end
      
      for k=1:numel(pxls)
        blk = pxls{k};
        ordr = 1:3;
        ordr([dim,3]) = ordr([3,dim]); % Swap dim with 3rd index
        blk = permute(blk,ordr);
        blk = blk(:,:,1:end-1);
        pxls{k} = ipermute(blk,ordr);
      end
    end
    
    function mtx = getDiffMtrx(prms, dim, mode)
      n_c = size(prms.ofsts,1);
      n_b = size(prms.ofsts,2);
      
      if VidRegion.sameBlk(prms)
        mtxs = makeBlkMtrx(1,1);
        mtx = SensingMatrixKron.constructKron({...
          SensingMatrixUnit(n_b*n_c), mtxs});
      else
        mtxs = cell(n_b, n_c);
        for iblk = 1:n_b
          for iclr = 1:n_c
            mtxs{iblk,iclr} = makeBlkMtrx(iclr,iblk);
          end
        end
        mtx = SensingMatrixBlkDiag.constructBlkDiag(mtxs(:));
      end
      
      function mt = makeBlkMtrx(iclr,iblk)
        sz = prms.blk_size(iclr,:,iblk);
        sz = sz(:)';
        mtb = {SensingMatrixUnit(sz(1)), SensingMatrixUnit(sz(2)), ...
          SensingMatrixUnit(sz(3))};
        
        switch mode
          case 0
            mt0 = SensingMatrixSelectRange.construct(1, sz(dim)-1, sz(dim));
            mt1 = SensingMatrixSelectRange.construct(2, sz(dim), sz(dim));
          case 1
            mt0 = SensingMatrixUnit(sz(dim));
            mt1 = SensingMatrixConcat({...
              SensingMatrixSelectRange.construct(2, sz(dim), sz(dim)),...
              SensingMatrixSelect.construct(1, sz(dim))});
          case 2
            mt0=SensingMatrixCascade.construct({...
              SensingMatrixSelectRange.construct(2,sz(dim),sz(dim),true),...
              SensingMatrixSelectRange.construct(1,sz(dim)-1, sz(dim),false)});
            mt1=SensingMtrixUnit(sz(dim));
          otherwise
            error('unexpected mode: %s', show_str(mode));
        end
        mtb{dim} = SensingMatrixCombine([-1,1], {mt0,mt1});

%         switch mode
%           case 0
%             mtr = SensingMatrixMatlab(sparse(...
%               [1:sz(dim)-1,1:sz(dim)-1], [1:sz(dim)-1, 2:sz(dim)],...
%               [-1*ones(1,sz(dim)-1,1), ones(1,sz(dim)-1,1)]));
%           case 1
%             mtr = SensingMatrixMatlab(sparse(...
%               [1:sz(dim),1:sz(dim)], [1:sz(dim), 2:sz(dim), 1],...
%               [-1*ones(1,sz(dim),1), ones(1,sz(dim),1)]));
%           case 2
%             mtr = SensingMatrixMatlab(sparse(...
%               [1:sz(dim),2:sz(dim)], [1:sz(dim), 1:sz(dim)-1],...
%               [ones(1,sz(dim),1), -1*ones(1,sz(dim)-1,1)]));
%           otherwise
%             error('unexpected mode: %s', show_str(mode));
%         end
%         if ~isequal(full(mtr.getMatrix()), full(mtb{dim}.getMatrix()))
%           error('matrices do not match');
%         end
          
        mt = SensingMatrixKron.constructKron(mtb(end:-1:1));
      end
    end
    
    function mtx = getDiff3Mtrx(prms, mode)
      n_c = size(prms.ofsts,1);
      n_b = size(prms.ofsts,2);
      
      if VidRegion.sameBlk(prms)
        mtxs = makeBlkMtrx(1,1);
        mtx = SensingMatrixKron.constructKron({...
          SensingMatrixUnit(n_b*n_c), mtxs});
      else
        mtxs = cell(n_b, n_c);
        for iblk = 1:n_b
          for iclr = 1:n_c
            mtxs{iblk,iclr} = makeBlkMtrx(iclr,iblk);
          end
        end
        mtx = SensingMatrixBlkDiag.constructBlkDiag(mtxs(:));
      end
      
      function mt = makeBlkMtrx(iclr,iblk)
          mtb = cell(3,1);
          sz = prms.blk_size(iclr,:,iblk);
          sz = sz(:)';
          for dim=1:3
            mtb = {SensingMatrixUnit(sz(1)), SensingMatrixUnit(sz(2)), ...
              SensingMatrixUnit(sz(3))};
            switch mode
              case 0
                mtb{dim} = SensingMatrixMatlab(sparse(...
                  [1:sz(dim)-1,1:sz(dim)-1], [1:sz(dim)-1, 2:sz(dim)],...
                  [-1*ones(1,sz(dim)-1,1), ones(1,sz(dim)-1,1)]));
              case 1
                mtb{dim} = SensingMatrixMatlab(sparse(...
                  [1:sz(dim),1:sz(dim)], [1:sz(dim), 2:sz(dim), 1],...
                  [-1*ones(1,sz(dim),1), ones(1,sz(dim),1)]));
              case 2
                mtb{dim} = SensingMatrixMatlab(sparse(...
                  [1:sz(dim),2:sz(dim)], [1:sz(dim), 1:sz(dim)-1],...
                  [ones(1,sz(dim),1), -1*ones(1,sz(dim)-1,1)]));
            end
          end
          mt = SensingMatrixKron.constructKron(mtb(end:-1:1));
      end
    end
    
    % Compute the matrix which performs a transform on one dimension in all
    % blocks and all colors of the region. The transform must be orthogonal
    % and with norm of 1.
    %   Input:
    %     prms - parameters specfying the blocks and colors offsets
    %     dim - the dimension on which the transform should work
    %     trnsfrm - the transform function
    %     inv_transform - the inverse transform function.
    function mtx = get1dTrnsfrmMtrx(prms, dim, trnsfrm, inv_trnsfrm)
      n_c = size(prms.ofsts,1);
      n_b = size(prms.ofsts,2);

      if VidRegion.sameBlk(prms)
        mtx = SensingMatrixKron.constructKron({SensingMatrixUnit(n_c*n_b) ...
          makeBlkMtrx(1,1)});
      else
        mtxs = cell(n_b, n_c);
        for iblk = 1:n_b
          for iclr = 1:n_c
            mtxs{iblk,iclr} = makeBlkMtrx(iclr,iblk);
          end
        end
        mtx = SensingMatrixBlkDiag.constructBlkDiag(mtxs(:));
      end
      
      function mt = makeBlkMtrx(iclr, iblk)
        sz = prms.blk_size(iclr,:,iblk);
        if sz(dim) > 1
          sz = sz(:)';
          mtb = {SensingMatrixUnit(sz(1)), SensingMatrixUnit(sz(2)), ...
            SensingMatrixUnit(sz(3))};
          mtb{dim} = SensingMatrixTrnsfrm(sz(dim), trnsfrm, inv_trnsfrm);
          mt = SensingMatrixKron.constructKron(mtb(end:-1:1));
        else
          mt = SensingMatrixUnit(prms.blk_len(iclr,:,iblk));
        end
      end
    end
    
    
    function same_blk = sameBlk(prms)
      % Return true if all color blocks are of the same size
      same_blk = true;
      n_c = size(prms.ofsts,1);
      n_b = size(prms.ofsts,2);
      % First check if all blocks are the same dimension
      for iblk=1:n_b
        for iclr=1:n_c
          if ~isequal(prms.blk_size(iclr,:,iblk), prms.blk_size(1,:,1))
            same_blk = false;
            return
          end
        end
      end
    end
    
    % Compute differences along one dimension
    %   Input:
    %     vec - Input video region. A a cell array of 3D matrices.
    %     dim_indx - dimension along which the differece is computed.
    %     mode - (optional). Sets the mode of the difference
    %            computation:
    %            0 - inside differnces - the output dimension is
    %                smaller by one from the input dimentsion.
    %            1 - circular differences - the output dimension
    %                is the same as the input dimesnion.
    %            2 - extended - zero is added before the entries,
    %                so the first entry is preserved.
    %  Output:
    %     dff - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays, but if mode = 0
    %           the size of the dim_indx dimension is smaller by 1.
    %           If d_vid_reg is missing
    %           dff is a vector (vectorized by d_vid_reg internally).
    %
    function dff = compDiff(vec, dim_indx, mode)
      if nargin < 4
        mode = 0;
      end
      
      switch(mode)
        case 0
          dff = VidRegion.compDiffInside(vec, dim_indx);
        case 1
          dff = VidRegion.compDiffCirc(vec, dim_indx);
        case 2
          dff = VidRegion.compDiffExtnd(vec, dim_indx);
      end
    end
    
    % Compute inside differences along one dimension.
    %   Input:
    %     vec - Input video region.  It is assumed to be a cell array
    %           of 3D matrices of pixel value of this region.
    %     dim_indx - dimension along which the differece is computed.
    %  Output:
    %     dff - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays, but the size of the
    %           dim_indx dimension is smaller by 1.
    %
    function dff = compDiffInside(vec, dim_indx)
      dff = cell(size(vec));
      for k = 1:numel(vec)
        blk = vec{k};
        dff{k}= diff(blk,1,dim_indx);
      end
    end
    
    % Compute circular differences along one dimension. This is the same as
    % compDiff, except that the another entry is added, of the first minus last
    % entries along the required dimension.
    %   Input:
    %     vec - Input video region.  It is assumed to be a
    %           pixel value of this region.
    %     dim_indx - dimension along which the differece is computed.
    %  Output:
    %     dff - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays
    %
    function dff = compDiffCirc(vec, dim_indx)
      dff = cell(size(vec));
      for k = 1:numel(vec)
        blk = vec{k};
        
        % Create a vector of corresponding to the dimensions of
        % blk, with 1 in the dim_indx entry and zeroes in all
        % other entires
        shift_vec = zeros(length(size(blk)),1);
        shift_vec(dim_indx)=-1;
        
        % Shift blk circularly by 1 along the dim_indx dimension
        sblk = circshift(blk,shift_vec);
        
        % compute the difference
        dff{k}= sblk - blk;
      end
    end
    
    % Compute extended differences along one dimension.
    %   Input:
    %     vec - Input video region. It is assumed to be a
    %           pixel value of this region.
    %     dim_indx - dimension along which the differece is computed.
    %  Output:
    %     dff - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays.
    %
    function dff = compDiffExtnd(vec, dim_indx)
      dff = cell(size(vec));
      for k = 1:numel(vec)
        blk = vec{k};
        
        % Insert zeroes at the beginning of each dim_indx
        % "row".
        sz = size(blk);
        sz(dim_indx)=1;
        blk = cat(dim_indx, zeros(sz), blk);
        dff{k}= diff(blk,1,dim_indx);
      end
    end
    
    % Apply the transpose of the difference along one dimension to a video
    % region of a size resulting from the transpose operation.
    %   Input:
    %     vec - Input video region.  If this is not a cell array, it is assumed
    %           to be a vectorized version of the data and it is pixelized
    %           before processing.  Otherwised it is assumed to be a
    %           pixel value difference of this region. Note that the video has
    %           one less pixel along the dim_indx dimension
    %     dim_indx - dimension along which the differece is computed.
    %     mode - (optional). Sets the mode of the difference
    %            computation:
    %            0 - inside differnces - the output dimension is
    %                smaller by one from the input dimentsion.
    %            1 - circular differences - the output dimension
    %                is the same as the input dimesnion.
    %            2 - extended - zero is added before the entries,
    %                so the first entry is preserved.
    %  Output:
    %     out - output vector
    %
    function out = compDiffTrnsp(vec, dim_indx, mode)
      if nargin < 4
        mode = 0;
      end
      switch(mode)
        case 0
          out = VidRegion.compDiffTrnspPxl(vec, dim_indx);
        case 1
          out = VidRegion.compDiffCircTrnspPxl(vec, dim_indx);
        case 2
          out = VidRegion.compDiffExtndTrnspPxl(vec,dim_indx);
      end
    end
    
    % Apply the transpose of the difference along one dimension to a video
    % region of a size resulting from the transpose operation.
    %   Input:
    %     vec - Input video region.  A a cell array of 3D matrices.
    %     dim_indx - dimension along which the differece is computed.
    %  Output:
    %     out - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays, but the size of the
    %           dim_indx dimension is smaller by 1.
    %
    function out = compDiffTrnspPxl(vec, dim_indx)
      out = cell(size(vec));
      for k=1:numel(vec)
        blk = vec{k};
        
        sz = size(blk);
        
        % Create a vector of corresponding to the dimensions of
        % blk, with 1 in the dim_indx entry and zeroes in all
        % other entires
        shift_vec = zeros(length(sz),1);
        shift_vec(dim_indx)=1;
        
        % Insert a zero slice at the end
        sz(dim_indx) = 1;
        blk = cat(dim_indx, blk, zeros(sz));
        
        % Shift blk circularly backward along the dim_indx dimension
        sblk = circshift(blk,shift_vec);
        
        % compute the difference
        out{k}= sblk - blk;
      end
    end
    
    % Apply the transpose of circular difference along one dimension to
    %  a video region of a size resulting from the transpose operation. This is
    %  the same as compDiffTrnspPxl except that the transform is circular.
    %   Input:
    %     vec - Input video region.  A a cell array of 3D matrices.
    %     dim_indx - dimension along which the differece is computed.
    %  Output:
    %     out - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays.
    %
    function out = compDiffCircTrnspPxl(vec, dim_indx)
      out = cell(size(vec));
      for k=1:numel(vec)
        blk = vec{k};
        % Create a vector of corresponding to the dimensions of
        % blk, with -1 in the dim_indx entry and zeroes in all
        % other entires
        shift_vec = zeros(length(size(blk)),1);
        shift_vec(dim_indx)=1;
        
        % Shift blk circularly backward along the dim_indx dimension
        sblk = circshift(blk,shift_vec);
        
        % compute the difference
        out{k}= sblk - blk;
      end
    end
    
    % Apply the transpose of the extended difference along one dimension to
    % a video region of a size resulting from the transpose operation.
    %   Input:
    %     vec - Input video region.  A a cell array of 3D matrices.
    %     dim_indx - dimension along which the differece is computed.
    %  Output:
    %     out - The output differences as a 2-dimensional cell array
    %           (color, block_no.) of 3 dimensional arrays, but the size of the
    %           dim_indx dimension is smaller by 1.
    %
    function out = compDiffExtndTrnspPxl(vec, dim_indx)
      out = cell(size(vec));
      for k=1:numel(vec)
        blk = vec{k};
        
        sz = size(blk);
        
        % Insert a zero slice at the end
        sz(dim_indx) = 1;
        blk = cat(dim_indx, blk, zeros(sz));
        
        out{k}= -diff(blk,1,dim_indx);
      end
    end
    
    % Compute a transform along one dimension
    %   Input:
    %     vec - Input video region. A cell array of 3D matrices.
    %     dim_indx - dimension along which the differece is computed.
    %     trnsfrm - A function handle which receives a 2 dim matrix and
    %           performs the transform along each column
    %  Output:
    %     out - The output DCT.
    %
    function out = comp1dTrnsfrm(vec, dim_indx, trnsfrm)
      out = cell(size(vec));
      for k=1:numel(vec)
        blk = vec{k};
        
        % Convert to a 2xdim matrix with dim_indx first
        sz = ones(1,3);
        szb = size(blk);
        sz(1:length(szb))=szb;
        tsz = sz;
        indices = 1:length(sz);
        if dim_indx ~= 1
          indices([dim_indx,1])=[1,dim_indx];
          tsz([dim_indx,1]) = sz([1,dim_indx]);
          blk = permute(blk, indices);
        end
        blk = reshape(blk(:), tsz(1), prod(tsz(2:end)));
        
        %perform the transform
        blk = trnsfrm(blk);
        
        % Convert back
        blk = reshape(blk(:), tsz);
        if dim_indx ~= 1
          blk = ipermute(blk, indices);
        end
        
        out{k} = blk;
      end
    end
    
    function out = do_multiDiffExtnd(inp, dim_cnt)
      % Perform Extended differences (compDiffExntd) several times along
      % several dimensions.
      % Input:
      %   inp - the input region as a cell array of 3D arrays.
      %   dim_cnt - a three dimensional array specifying how many times
      %        to compute the difference along each dimension
      %   blk_stt - Status of the input
      % Output:
      %   out - output
      for k = 1:length(dim_cnt)
        for m=1:dim_cnt(k)
          out = VidRegion.compDiffExtnd(inp, k);
        end
      end
    end
    
    function out = do_multiDiffUnExtnd(inp, dim_cnt)
      % Undo the operatrion of multiDiffExtnd.
      % Input:
      %   inp - the input region pixels as a cell array of 3D matrices.
      %   dim_cnt - a three dimensional array specifying how many times
      %        to compute the difference along each dimension
      % Output:
      %   out - output
      out = inp;
      for k = length(dim_cnt):-1:1
        for m=1:dim_cnt(k)
          for iblk = 1:numel(inp)
            out{iblk} = cumsum(out{iblk}, dim_indx);
          end
        end
      end
    end
  end
  
  methods (Access=protected)
    
    function setBlksLength(obj)
      Cblk_len = prod(obj.clrBlkSize());
      obj.vec_len = obj.n_blk * obj.n_color * Cblk_len;
      obj.ext_vec_len = obj.n_blk * obj.n_color * obj.blkr.ext_Cblk_len;
      
      sz = [obj.n_color, 3, obj.n_blk,obj.blkr.N_BLK_STT];
      obj.stt_Cblk_size = zeros(sz);
      
      for k=1:obj.n_blk
        [~,~,blk_len,~,~,~] = obj.blkr.blkPosition(obj.blk_indx(k,:));
        
        obj.stt_Cblk_size(:,:,k,obj.blkr.BLK_STT_RAW) = blk_len;
        obj.stt_Cblk_size(:,:,k,obj.blkr.BLK_STT_INTRPLT) = ...
          ones(obj.n_color,1)*blk_len(1,:);
        obj.stt_Cblk_size(:,:,k,obj.blkr.BLK_STT_EXTND) = ...
          ones(obj.n_color,1)*obj.blkr.blk_size(1,:);
        obj.stt_Cblk_size(:,:,k,obj.blkr.BLK_STT_ZEXT) = ...
          ones(obj.n_color,1)*obj.blkr.ext_Cblk_size;
      end
      
      obj.stt_Cblk_size(:,:,:,obj.blkr.BLK_STT_WNDW) = ...
        obj.stt_Cblk_size(:,:,:,obj.blkr.BLK_STT_EXTND);
      
      obj.stt_Cblk_len = prod(obj.stt_Cblk_size,2);
      obj.stt_Cblk_len = reshape(obj.stt_Cblk_len(:),sz([1,3,4]));
      obj.stt_blk_len = sum(obj.stt_Cblk_len,1);
      obj.stt_blk_len = reshape(obj.stt_blk_len(:), sz([3,4]));
      obj.stt_vec_len = sum(obj.stt_blk_len,1);
      
      ofst = cumsum(reshape(obj.stt_Cblk_len,...
        obj.n_blk*obj.n_color,obj.blkr.N_BLK_STT),1);
      ofst = [zeros(1,obj.blkr.N_BLK_STT); ofst(1:end-1,:)];
      obj.stt_Cblk_offset = ...
        reshape(ofst(:), obj.n_color, obj.n_blk, obj.blkr.N_BLK_STT);

%       bl = permute(obj.stt_Cblk_len,[2,1,3]);
%       ofst = cumsum(reshape(bl,obj.n_blk*obj.n_color,obj.blkr.N_BLK_STT),1);
%       ofst = [zeros(1,obj.blkr.N_BLK_STT); ofst(1:end-1,:)];
%       ofst = reshape(ofst(:), obj.n_blk, obj.n_color, obj.blkr.N_BLK_STT);
%       obj.stt_Cblk_offset = ipermute(ofst, [2,1,3]);
    end
    
  end
  
  methods (Access = private)
    % Compute pxl_cnt_map and related variales
    compPxlCntMap(obj)
    
    compExpPxlMat(obj)
    compUnExpPxlMat(obj)
    
  end
end

