classdef VidBlocker <  CompMode
  %VidBlocker objects contains operator to divide video into equal size blocks
  %and to reconstruct it back from those blocks.
  %   Detailed explanation goes here
  properties (Constant)
    
    vid_info_flds = {'vid_size_v', 'vid_size_h', 'vid_size_t', ...
      'uv_ratio_v', 'uv_ratio_h', 'uv_ratio_t', 'fps';...
      'I','I','I','I','I','I','F'};
    
    blk_info_flds = {'indx_v', 'indx_h', 'indx_t',...
      'ofst_v', 'ofst_h', 'ofst_t',...
      'len_v', 'len_h', 'len_t',...
      'ovlp_b_v', 'ovlp_b_h', 'ovlp_b_t',...
      'ovlp_f_v', 'ovlp_f_h', 'ovlp_f_t',...
      'yw_type';...
      'I','I','I', ...
      'I','I','I', ...
      'I','I','I', ...
      'I','I','I', ...
      'I','I','I',...
      'I'...
      };
    
    % Processing state of a video block. These are sequential steps in
    % the processing of a block
    BLK_STT_RAW = 1; % Raw block - no processing
    BLK_STT_INTRPLT = 2; % UV components have been interpolated to size of Y
    BLK_STT_EXTND = 3; % edge blocks have been extended to full size
    BLK_STT_WNDW = 4; % Block has been windowed.
    BLK_STT_ZEXT = 5; % Blocks have been zero extended and wrap extended
    N_BLK_STT = VidBlocker.BLK_STT_ZEXT;
    
    % Indices for indicating block position in a particular dimension
    FST=1; % First block
    MID=2; % Middle
    LST=3; % Last
  end
  
  properties
    % RawVidInfo object describing the raw video file
    vid_info = [];
    
    % The ratio between the size in the Y component and the sizes in
    % the UV componensts. Empty if luminance only
    uv_ratio = [];
    
    % An array of n_clr by 3 of the sizes of the raw video in each
    % component.
    vid_size = [];
    
    % Frames per second
    fps;
    
    % Block size. each row is (height, width, time) of one component.
    % If the block is black and white there is one row (Y), otherwise
    % there are 3 rows (YUV). blk_size(k,:) for k>1 must be divisible
    % by uv_ratio
    blk_size=[];
    
    % Block overlap. Each row is the overlap (height, width, time) of
    % one component. Number of rows is same as blk_size. ovrlp(k,:)
    % for k>1 must be divisible by uv_ratio
    ovrlp=[];
    
    wnd_enc_type=[];     % type of window for encoding
    wnd_dec_type=CS_DecParams.WND_TRIANGLE;     % type of window for decoding
    
    % zero extension of a block.  It is a 2x3 array, where
    % the first row is the backward extnesion and second is
    % the forward extension.  Columns represent (H,V,T).
    zext = [0 0 0; 0 0 0];
    
    % Wrap extension of a block
    wext = [0 0 0];
    
    % Various sizes and lengths
    blk_cnt =[]; % Number of blocks in each dimesnion
    Cblk_cnt=[]; % Number of color blocks in each dimesnion (incl. color).
    Rblk_size = []; % size of raw block (position, clr, dim)
    Rblk_ofst = []; % Offset of actual video beginning in the block 
                    % (position, clr, dim)
    Iblk_size = []; % size of interpolated block (position,dim)
    ext_Cblk_len=0;% Block length including zero extension
    ext_Cblk_size = []; % Block size including zero extension
    
  end
  
  properties (Access = private)
    wnd_enc=[];         % window for encoding
    wnd_dec=[];         % window for decoding
    
    % a 3D matrix of the windowing weights in each pixel of a block
    wnd_enc_blk_wgts = [];
    
    % A struct array used for conversion between different states.
    % Each entry in the struct array contains the following possible
    % fields:
    %   M - the matrix in the entry (from, to, pos, clr, dim) performs the
    %       expansion (or un-expansion) of a block from state 'from' to 
    %       state 'to', along the dimension dim, where the block is of
    %       color clr and in position (in the dim dimension) of pos (FST,
    %       MID, LST).
    %   U,L,V - The SVD of M. M=U*L*V', where L is diagonal and the columns 
    %       of U and rows of V are orthonormal.
    %   I - pseudo inverese of M. 
    %   R - V*LI, LI= pseudo inverse of L
    % The matrix in the entry (from, to, pos, clr, dim) performs the
    % expansion (or un-expansion) of a block from state 'from' to state
    % 'to', along the dimension dim, where the block is of color clr
    % and in position (in the dim dimension) of pos (FST, MID, LST).
    expnd_dim = [];
    
    % Same as expnd_dim but for whole colors, thus it is indexed by 
    % (from,to,v_pos,h_pos,t_pos,clr)
    expnd_clr = [];

      % Same as expnd_clr but for whole blocks, thus it is indexed by
    % (from,to,v_pos,h_pos,t_pos)
    expnd_blk = [];
  end
  
  methods
    function obj = VidBlocker(params, raw_vid, opts)
      % Constructor
      % Input (all parameters are optional)
      %   params - Can be one of the following: 
      %     A VidBlocker object, in which case no other arguments are
      %       allowed and this object is constructed as a copy of params;
      %     [] - params is computed from raw_vid and opts (which must be
      %          present)
      %     A struct of parameters which may contain the
      %       following fields (default options, if any are in <angle brackets>).
      %         yb_size - values for Y block size (V,H,T)
      %         ovrlp  - overlap of the Y block <[0 0 0]>;
      %         w_type_e - value of encoder wnd_type [obj.wnd_enc_type]
      %         w_type_d - value of decoder wnd_type [obj.wnd_dec_type]
      %         monochrom - if true, force monochromatic processing [false]
      %         zero_ext - optional specification for zext,
      %               [zero_ext_b; zero_ext_f].  If it has 3 rows the
      %               the third row is wrap_ext. It can also be a struct
      %               with fields 'b','f','w' specifying zero_ext_b,
      %               zero_ext_f and wrap_ext. [obj.zext]
      %         wrap_ext - optional argument for wext (cannot be present if
      %               wrap_ext was specified in zero_ext).
      %         linient - normally if yb_size is divisible by
      %               UVrat, an error occurs.  If linient is true we
      %               divide and truncate [false].
      %         use_gpu - logical indicating whether to use the GPU
      %         use_signle - logical indicating whether to single precision
      %   raw_vid - A RawVidInfo object describing the RawVideo. This
      %     parameter should be omitted if params is a VidBlocker object
      %   opts - (optional) A CS_EncVidParams object. If present, all options
      %          of params, except for linient and w_type_d, are overriden
      %          by opts.

      if isa(params, 'VidBlocker')
        if nargin > 1
          error(['If the first argument is a VidBlocker, no other argument'
            ' should be speciffied']);
        end
        [params, raw_vid] = params.getParams();
      else
        if isempty(params)
          if nargin <= 2
            error('opts not specified while params is empty');
          end
          params = struct();
        end
        if nargin > 2
          params = VidBlocker.compParamsFromOpts(params, raw_vid, opts);
        end
        params = obj.setParamsDefaults(params, raw_vid);
      end
      
      obj.setCastIndex();
      obj.setCastFloat();
      
      obj.init(params, raw_vid);
    end
    
    function eql = isEqual(obj, other)
      if ~strcmp(class(obj), class(other)) || ...
          ~isequal(size(obj),size(other))
        eql = false;
        return;
      elseif all(eq(obj,other))
        eql = true;
        return
      end
      
      mc = metaclass(obj);
      props = mc.PropertyList;
      for k=1:length(props)
        if props(k).Constant || strcmp(props(k).GetAccess,'private')
          continue;
        end
        prop = props(k).Name;
        
        if ~isEqual(obj.(prop), other.(prop))
          eql = false;
          return;
        end
        
      end
      eql = true;

    end
    
    
    % Get the number of pixels in a clr block (including extension)
    function n_pxl = ttlPxlClrBlk(obj)
      n_pxl = prod(obj.blk_size(1,:));
    end
    
    % Return information about the video as a whole in the form of a
    % struct.
    function vid_info = getVidInfo(obj)
      if isempty(obj.uv_ratio)
        uvr = [0, 0, 0];
      else
        uvr = obj.uv_ratio;
      end
      
      vid_info = struct(...
        'vid_size_v', obj.vid_size(1,1),...
        'vid_size_h', obj.vid_size(1,2),...
        'vid_size_t', obj.vid_size(1,3),...
        'uv_ratio_v', uvr(1),...
        'uv_ratio_h', uvr(2),...
        'uv_ratio_t', uvr(3),...
        'fps', obj.fps);
    end
    
    % Return information about single block in the form of a struct
    % Input
    %   obj - this object
    %   v - vertical block index
    %   h - horizontal block index
    %   t - temporal block index
    % Output
    %   blk_info -a struct with fields specified by blk_info_flds
    function blk_info = getBlkInfo(obj,v,h,t)
      if nargin == 2
        indx = v;
      else
        indx = [v,h,t];
      end
      [offset, ~, len, ~, ~, blk_ofst] = obj.blkPosition(indx);
      blk_info = cell(1,size(obj.blk_info_flds,2));
      blk_info = cell2struct(blk_info,obj.blk_info_flds(1,:),2);
      dm = {'_v','_h','_t'};
      for k=1:3
        blk_info.(['indx' dm{k}]) = indx(k);
        blk_info.(['ofst' dm{k}]) = offset(1,k)-1;
        blk_info.(['len' dm{k}]) = len(1,k);
        blk_info.(['ovlp_b' dm{k}]) = obj.ovrlp(1,k) - blk_ofst(1,k);
        blk_info.(['ovlp_f' dm{k}]) = obj.ovrlp(1,k) + ...
          obj.blk_size(1,k) - blk_ofst(1,k) - len(1,k);
      end
      blk_info.w_type_e = obj.wnd_enc_type;
      blk_info.w_type_d = obj.wnd_dec_type;
    end
    
    % Get a block with a uniform content
    %  Input
    %    obj - this object
    %    val - value to assign to all pixels (devault = 0)
    %  Output
    %    blk - Array of size blk_size(1,:)
    function blk = getUniformBlk(obj, val)
      if nargin < 2
        blk  = zeros(obj.blk_size(1,:));
      else
        blk  = ones(obj.blk_size(1,:)) * val;
      end
    end
    
    % Extract one block from the video raw_vid.
    % blk_indx is the index of the block in raw_vid.
    %   Input:
    %     obj - this object
    %     raw_vid - raw video. A cell array of 1 (Y) or 3 (YUV) cells.
    %     blk_indx - index of block in video
    %     stt - required state of video in the block (BLK_STT_xxx)
    %  Output:
    %     blk - Output blk. A cell array of 1 (Y) or 3 (YUV) cells.
    %           first and last blocks are extended with same values.
    %           UV components are interpolated to the same size as Y.
    function blk = getSingleBlk(obj, raw_vid, blk_indx, stt)
      % Get raw vector
      [orig, blk_end, ~, ~, ~, ~] = obj.blkPosition(blk_indx);
      vec = do_getSingleRawBlk(obj, raw_vid, orig, blk_end);
      
      % expand to state stt
      xpnd = obj.getExpandMtrx(obj.BLK_STT_RAW, stt, blk_indx, 0);
      vec = xpnd.M.toFloat(vec);
      blk = xpnd.M.multVec(vec);
     end
    
    % Put a single block into raw_vid.
    function raw_vid = putSingleBlk(obj, blk, raw_vid, blk_indx, blk_stt)
      [orig, blk_end, blen, is_first, is_last, blk_ofst] =...
        obj.blkPosition(blk_indx);
      
      % Skip empty blocks
      if isempty(blk) || isempty(blk{1})
        return
      end

      % If the block has been windowed but not with the right window,
      % or if it is a first or last block, which is windowed on
      % encoding but not on decoding, the windowing has to be undone
      if blk_stt == obj.BLK_STT_WNDW && ...
          (~isequal(obj.wnd_enc_type, obj.wnd_dec_type) ||...
          any((is_first|is_last) & obj.ovrlp(1,:)))
        blk = obj.unWindowBlk(blk, obj.wnd_enc, [0,0,0], [0,0,0]);
        blk_stt = obj.BLK_STT_EXTND;
      end
      
      % If windowing is needed and the block is not in BLK_STT_EXTND,
      % it needs to be brought to this state
      if blk_stt < obj.BLK_STT_WNDW && any(obj.ovrlp(:))
        if blk_stt == obj.BLK_STT_RAW
          blk = obj.interpolate(blk);
          blk_stt = obj.BLK_STT_INTRPLT;
        end
        if blk_stt == obj.BLK_STT_INTRPLT;
          blk = obj.extendBlk(blk, blk_ofst(1,:));
          blk_stt = obj.BLK_STT_EXTND;
        end
        if blk_stt == obj.BLK_STT_EXTND
          blk = obj.windowBlk(blk, obj.wnd_dec, is_first, is_last);
        end
      elseif blk_stt == obj.BLK_STT_WNDW
        blk_stt = obj.BLK_STT_EXTND;
      end
      
      if blk_stt == obj.BLK_STT_EXTND
        blk = obj.undoExtendBlk(blk, blk_ofst(1,:), blen(1,:));
        
        % Note that we ignore the decoding window.
        blk_stt = obj.BLK_STT_INTRPLT;
      end
      if  blk_stt == obj.BLK_STT_INTRPLT
        % Un-interpolation is done on the windowed signa.
        blk = obj.undoInterpolate(blk);
      end
      
      for k=1:size(obj.blk_size,1)
        blk_val = blk{k};
        
        raw_vid{k}(orig(k,1):blk_end(k,1), orig(k,2):blk_end(k,2),...
          orig(k,3):blk_end(k,3)) =...
          raw_vid{k}(orig(k,1):blk_end(k,1), orig(k,2):blk_end(k,2),...
          orig(k,3):blk_end(k,3)) + blk_val;
      end
    end
    
    
    % Extract one or more blocks from the video raw_vid.
    % Input:
    %    obj - this object
    %    blk_indx - The blocks to be read. can be empty of have
    %               1,2,or,3 entries. If it has 3 entries, one block is
    %               read, whose index is blk_indx. If it has 2 entries
    %               a column of blocks is read whos H,T indices are
    %               blk_indx.  If it has 1 entry, a matrix of blocks is
    %               read whose T index is blk_indx. If blk_indx is empty
    %               all the blocks in raw_vid are extracted.
    %    raw_vid - source of raw video, passed to getSingleBlk()
    %    stt - required state of video in the block (BLK_STT_xxx)
    % Output:
    %    blk - a cell array of (C,V,H,T) (component, vertical,
    %          horizontal, time) where each cell contains
    %          a three dimensional array of pixels.
    %    nxt_blk_indx is of the same dimension as blk_indx and is the
    %         position of the next block or group of blocks.
    function [blk, nxt_blk_indx] = getBlk(obj, raw_vid, blk_indx, stt)
      len_blk_indx = length(blk_indx);
      if len_blk_indx < 3
        cnt = obj.blk_cnt(3-len_blk_indx);
        blk = cell(cnt,1);
        for k=1:cnt
          blk{k} = obj.getBlk(raw_vid, [k, blk_indx], stt);
        end
        blk = vertcat(blk{:});
        
%         blk = cell(obj.blk_cnt(1:end-len_blk_indx));
%         switch len_blk_indx
%           case 0
%             for k=1:obj.blk_cnt(3)
%               blk0 = obj.getBlk(raw_vid, [k blk_indx], stt);
%               if isempty(blk0)
%                 blk = [];
%                 break;
%               end
%               blk(:,:,k) = blk0;
%             end
%           case 1
%             for k=1:obj.blk_cnt(2)
%               blk0 = obj.getBlk(raw_vid, [k blk_indx], stt);
%               if isempty(blk0)
%                 blk = [];
%                 break;
%               end
%               blk(:,:,k) = blk0;
%             end
%           case 2
%             for k=1:obj.blk_cnt(1)
%               blk0 = obj.getBlk(raw_vid, [k blk_indx], stt);
%               if isempty(blk0)
%                 blk = [];
%                 break;
%               elseif ~iscell(blk0)
%                 blk0 = {blk0};
%               end
%               blk(:,k) = blk0;
%             end
%         end
      else
        blk = obj.getSingleBlk(raw_vid, blk_indx, stt);
        if isempty(blk)
          blk = [];
        end
      end
      
      if nargout >= 2
        if isempty(blk)
          nxt_blk_indx = [];
        else
          nxt_blk_indx = obj.calcNxtIndx(blk_indx);
        end
      end
    end
    
    % Put one or more blocks blks into the raw video array raw_vid.
    % blks is the array of blocks to insert. It can be a single block,
    % a column of blocks,a matrix of blocks (V,H) or a 3 dimnsional array
    % of blocks. raw_vid (optional) is the raw video into which the blocks
    %  need to be put. If not given it will be created.
    % blk_indx (optional) is the position in which the block needs to
    % be put. It can be missing only if raw_vid is missing.  Otherwise,
    % if it has 3 entries, blks has to have a single entry indicated by
    % blk_indx.  If it has 2 entries, blks has to be a column of blocks
    % whose position is givenby blk_indx.  If it has 1 entry the blks
    % should be a matrix of blocks (V,H) with temporal position
    % indicated by blk_indx. If blk_indx is empty then blks should be
    % three dimensional and fill raw_vid
    %
    % The function returns the updated raw vid
    % nxt_blk_indx is of the same dimension as blk_indx and is the
    % position of the next block or group of blocks.
    function [raw_vid, nxt_blk_indx] = putBlk(obj, blks, blk_stt, raw_vid,...
        blk_indx)
      n_clr = size(obj.blk_size, 1);
      
      if nargin < 5
        if nargin == 4
          error('blk_indx cannot be missing when raw_vid is specified');
        else
          raw_vid = cell(n_clr,1);
          for k = 1:n_clr
            raw_vid{1} = zeros(size(blks(2:end)) .* ...
              (obj.blk_size(k,:)-obj.ovrlp(k,:)) + obj.ovrlp(k,:));
          end
          blk_indx = [];
        end
      end
      
      len_blk_indx = length(blk_indx);
      
      if len_blk_indx < 3
        switch len_blk_indx
          case 0
            for k=1:size(blks,4)
              raw_vid = obj.putBlk(blks(:,:,:,k), blk_stt,...
                raw_vid, [k blk_indx]);
            end
          case 1
            for k=1:size(blks,3)
              raw_vid = obj.putBlk(blks(:,:,k), blk_stt,...
                raw_vid, [k blk_indx]);
            end
          case 2
            for k=1:size(blks,2)
              raw_vid = obj.putBlk(blks(:,k), blk_stt,...
                raw_vid, [k blk_indx]);
            end
        end
      else
        raw_vid = obj.putSingleBlk(blks, raw_vid, blk_indx, blk_stt);
      end
      
      nxt_blk_indx = obj.calcNxtIndx(blk_indx);
    end
    
    % Write arrays of blocks into a file
    % Input
    %   obj - this object
    %   outfile - a file name (string) or a file handle. If  empty, or
    %             numeric and equals -1, no writing is done.
    %   blks - an array of 4 dimensions - color, height, width, time
    %          (frame).  Missing upper indices are taken
    %          as 1.
    %   blk_stt - processing state of the blocks (BLK_STT_...)
    %   frm_ovrlp - (optional) If not empty it is a cell array of residual
    %            video corresponding to temporal overlay. One cell for
    %            each color, where the number of frames in each color
    %            is obj.ovrlp(iclr,3).
    %   is_diff (optional) - if true, data is a difference and will be
    %          scaled and centered. Default, false.
    % Output
    %   nfr - If successful, number of frames written out.  otherwise
    %         an error message
    %   vid - returns the output videos.  If outfiles is a cell array
    %         it is a cell array of videos.  Otherwise it is a single
    %         video, where a video is a cell array of 1 (black and
    %         white) or 3 (YUV) arrays.
    %   frm_ovrlp - Residual video for next writing. Contains cells with
    %               3-dimensional blocks where the
    %               third dimension is obj.ovrlp(1,3). If missing all
    %               cells will be written
    function [nfr, vid, frm_ovrlp] = writeBlocks(obj, outfile, blks, ...
        blk_stt, frm_ovrlp, is_diff)
      
      if nargin < 6
        is_diff = false;
        if nargin < 5
          frm_ovrlp = [];
        end
      end
      inp_ovrlp = frm_ovrlp;
      if obj.ovrlp(1,3)==0 || isempty(inp_ovrlp)
        nfr = size(blks,4)*(obj.blk_size(1,3) - obj.ovrlp(1,3));
        vid = obj.vid_info.createEmptyVideo(nfr);
        vid = obj.putBlk(blks(:,:,:,:), blk_stt, vid, []);
      else
        nfr = (size(blks,4)+1)*(obj.blk_size(1,3) - obj.ovrlp(1,3));
        vid = obj.vid_info.createEmptyVideo(nfr);
        % Add overlap frames
        for iclr = 1:min(length(inp_ovrlp),size(blks,1))
          frm_stp = obj.blk_size(iclr,3)- obj.ovrlp(iclr,3);
          vid{iclr}(:,:,frm_stp-obj.ovrlp(iclr,3)+1: frm_stp) =...
            inp_ovrlp{iclr};
        end
        
        %insert blocks
        for k=1:size(blks,4)
          % Use of k+1 indicates to putBlk that this is not
          % temporally first, hence it should not be truncated.
          vid = obj.putBlk(blks(:,:,:,k), blk_stt, vid, k+1);
        end
        
        % Remove filler frames
        for iclr=1:length(vid)
          bgn = obj.blk_size(iclr,3)- 2*obj.ovrlp(iclr,3) + 1;
          vid{iclr} = vid{iclr}(:,:,bgn:end);
        end
        nfr = nfr - (obj.blk_size(1,3)-2*obj.ovrlp(1,3));
      end
      
      % Generate output overlap
      if obj.ovrlp(1,3)
        if nargout >= 3
          frm_ovrlp = obj.vid_info.createEmptyVideo(obj.ovrlp(1,3));
          for iclr = 1:length(vid)
            frm_ovrlp{iclr} = vid{iclr}(:,:,end-obj.ovrlp(iclr,3)+1:end);
          end
        end
        
        % Remove end overlap from vid
        for iclr=1:length(vid)
          vid{iclr} = vid{iclr}(:,:,1:end-obj.ovrlp(iclr,3));
        end
        nfr = nfr - obj.ovrlp(1,3);
      else
        frm_ovrlp = [];
      end
      
      if is_diff
        for iclr = 1:size(blks,1)
          vid{iclr} = 0.5*(vid{iclr}+obj.vid_info.getPixelMax());
        end
      end
      
      % check if writing out is necessary
      if ~(isempty(outfile) || (isnumeric(outfile) && outfile==-1))
        err_msg = write_raw_video(outfile, vid, obj.vid_info);
        if ~isempty(err_msg)
          nfr = err_msg;
        end
      end
    end
    
    function nfrm = nRawFrmsInBlk(obj, blk_indx)
      % Returns the number of raw frames in block indexed by blk_indx
      is_first = (blk_indx(3) == 1);
      is_last = (blk_indx(3) == obj.blk_cnt(3));
      pos = 1 + double(~is_first) + double(is_last);
      nfrm = obj.Rblk_size(pos,1,3);
    end
    
    % Provides information about a block position in the raw video
    % volume.
    % input:
    %   obj - this object
    %   blk_indx - index of block (V,H,T)
    % output:
    %   origin - lowest coordingates of block in raw video
    %   blk_end - highest coordinates of block in raw video
    %   blk_len - size of the subblock  containing original pixels - the
    %             nominal block size minus overlap outside the raw video.
    %   is_first - a row vector with logical values indication whether
    %             this is a first block in each dimension
    %   is_last  - a row vector with logical values indication whether
    %             this is a last block in each dimensin
    %   blk_ofst - Offset to start of original video pixels in the block.
    %              May be non-zero only if block is first.
    %
    function [origin, blk_end, blk_len, is_first, is_last, blk_ofst] =...
        blkPosition(obj, blk_indx)
      is_first = (blk_indx == 1);
      is_last = (blk_indx == obj.blk_cnt);
      pos = 1 + double(~is_first) + double(is_last);
      n_clr = size(obj.blk_size,1);
      blk_len = zeros(n_clr,3);
      blk_ofst = zeros(n_clr,3);
      for k=1:3
        blk_len(:,k) = obj.Rblk_size(pos(k),:,k);
        blk_ofst(:,k) = obj.Rblk_ofst(pos(k),:,k);
      end
      
      b_sz = obj.blk_size - obj.ovrlp;
      start = (ones(n_clr,1) * (blk_indx-1)) .* b_sz + 1;
      % correct for overlap
      origin = start - reshape(obj.Rblk_ofst(obj.FST,:,:),n_clr,3) + blk_ofst;
      blk_end = origin + blk_len -1;
    end
    
    % Compute position indices (FST, MID, LST) for each dimension in a
    % group of blocks
    %   Input:
    %     obj - this object
    %     blk_indx - an array whose rows are block indices (V,H,T)
    %   Output:
    %     an array liek blk_indx, but each entry is replaced by one of
    %     FST,MID,LST.
    function pos = indexPosition(obj, blk_indx)
      n_clr = size(obj.blk_size, 1);
      pos = blk_indx;
      for ib=1:size(pos,1)
        for dim =1:3
          if blk_indx(ib,dim) == 1 && ...
              any(obj.Rblk_ofst(obj.FST,1:n_clr,dim))
            pos(ib,dim) = obj.FST;
          elseif blk_indx(dim) == obj.blk_cnt(dim) && any(...
              obj.Rblk_size(obj.LST,1:n_clr,dim) < obj.blk_size(1:n_clr,dim)')
            pos(ib,dim) = obj.LST;
          else
            pos(ib,dim) = obj.MID;
          end
        end
      end
    end
    
    function n_pxl = nOrigPxlInBlk(obj, blk_indx)
      % Compute the number of original pixels in a block (not including zero
      % extension and interpolation, but including all colors). 
      % overlap pixels count is
      % divided by the number of blocks with which they are shared.
      % input:
      %   obj - this object
      %   blk_indx - index of block (V,H,T)
      % Output
      %   n_pxl - number of pixels in the block
      
      [~,~,blk_len, is_first, is_last,~] = obj.blkPosition(blk_indx);
      rt = obj.vid_info.intrpltRatio();
      blk_len = blk_len - ...
        (ones(size(obj.ovrlp,1),1)*(0.5*(2-(is_first+is_last)))).*...
        obj.ovrlp;
      nc_pxl = prod(blk_len,2);
      nc_pxl(1) = nc_pxl(1)/rt.Y;
      nc_pxl(2:end) = nc_pxl(2:end)/rt.UV;
      n_pxl = sum(nc_pxl);
    end
    
    % Get the pre-encode windowing weights for one color block.
    %   Input:
    %   blk_indx - index of block (V,H,T)
    function wgts = getWindowCBlkWgts(obj)
      if isempty(obj.wnd_enc_blk_wgts)
        sz = obj.blk_size(1,:);
        lsz = prod(sz);
        indcs = (1:lsz)';
        wgts = reshape(ones(size(indcs)), sz);
        obj.wnd_enc_blk_wgts = obj.windowBlk(wgts, obj.wnd_enc, [0,0,0], [0,0,0]);
      end
      wgts = obj.wnd_enc_blk_wgts;
    end
    
    function [params, raw_vid] = getParams(obj)
      params = struct(...
        'yb_size', obj.blk_size(1,:),...
        'w_type_e', obj.wnd_enc_type,...
        'w_type_d', obj.wnd_dec_type,...
        'ovrlp', obj.ovrlp(1,:),...
        'monochrom', isempty(obj.uv_ratio),...
        'fps', obj.fps,...
        'zero_ext', obj.zext,...
        'wrap_ext', obj.wext...
        );
      raw_vid = obj.vid_info.copy();
    end
    
    % Compute matrices which expand (or unexpand) a vector of video at
    % a given state into other states. The matrix saves the components of
    % the computed matrices, so real computation is done only once.
    % Input:
    %   obj - this object
    %   from - state of input (obj.BLK_STT_x)
    %   to - state of output (obj.BLK_STT_x)
    %   blk_indx - [h v t] block address. Internally this function may be
    %               also called with a scalar blk indx which can be
    %               obj.FST,obj.MID or obj.LST
    %   eps - Epsilon to use in computing SVD
    %   clr - If present, the matrix is specific for a color. Otherwise,
    %         the matrix is for a vector of all colors.
    %   dim - dimension (1-3). If present, the matrix computes the
    %         expansion (or unexpansion) only along the specified
    %         dimension. Otherwise it computes it along all dimensions.
    % Output
    %   xpnd - a struct with component 'M', 'U', 'L', 'V', 'R', 'I' which are objects
    %          of type SensingMatrix. xpnd.M is the required matrix. The 
    %          'U', 'L' and 'V' fields are the singular value decomposition
    %          (SVD) of xpnd.M. Thus:
    %            xpnd.M = xpnd.U * xpnd.L * xpnd.V'
    %          where xpnd.L is diagonal (of type SensingMatrixDiag), and the
    %          columns of xpnd.U and xpnd.V are orthonormal . 
    %          xpnd.R is the inverse of the matrix xpnd.L * xpnd.V', that
    %          is, xpnd.R = xpnd.V * inv(xpnd.L);
    %          xpnd.I is the pseudo inverse of xpnd.M
    function xpnd = getExpandMtrx(varargin)
      obj = varargin{1};
      if nargin < 5
        eps = 0;
      else
        eps = varargin{5};
      end
      if nargin < 6
        clr = [];
      else
        clr = varargin{6};
      end
      if nargin < 7
        dim = [];
      else
        dim = varargin{7};
      end
      
      from = varargin{2};
      to = varargin{3};
      do_copy = true;
      pos = obj.cmpPos(varargin{4},clr);
      switch nargin
        case 7
          xpnd = obj.expnd_dim(from,to, pos(dim), clr, dim);
          if ~isempty(xpnd.M) && ~isempty(xpnd.U)
            do_copy = false;
          end
        case 6
          xpnd = obj.expnd_clr(from,to, pos(1), pos(2), pos(3), clr);
          if ~isempty(xpnd.M) && ~isempty(xpnd.U)
            do_copy = false;
          end
        otherwise
          xpnd = obj.expnd_blk(from,to, pos(1), pos(2), pos(3));
          if ~isempty(xpnd.M) && ~isempty(xpnd.U)
            do_copy = false;
          end
      end
      if do_copy
        xpnd = do_getExpandMtrx(varargin{:});
      end
      if isempty(xpnd.R)
        [LI,UI,VI] = SensingMatrix.invertSVD(xpnd.L, xpnd.U, xpnd.V, eps);
        xpnd.I = SensingMatrix.constructSVD(LI,UI,VI);
        xpnd.I.use_gpu = obj.use_gpu;
        xpnd.I.use_single = obj.use_single;
        xpnd.R = SensingMatrixCascade.constructCascade({UI,LI});
        xpnd.R.use_gpu = obj.use_gpu;
        xpnd.R.use_single = obj.use_single;
        
%         % Test code        
%         XR = obj.calcExpandR(xpnd, eps);
%         if ~isequal(XR,xpnd.R)
%           error('xpnd.R and XR are not equal');
%         end

        do_copy = true;
      end
      
      if do_copy
        switch nargin
          case 7
            if from >= obj.BLK_STT_EXTND && to >= obj.BLK_STT_EXTND
              obj.expnd_dim(from, to, :, :, dim) = xpnd;
            elseif from >= obj.BLK_STT_INTRPLT && to >= obj.BLK_STT_INTRPLT
              for ps=1:3
                if obj.Rblk_size(ps,clr,dim) == ...
                    obj.Rblk_size(pos(dim), clr, dim) &&...
                    obj.Rblk_ofst(ps, clr, dim) ==...
                    obj.Rblk_ofst(pos(dim), clr, dim)
                  
                  obj.expnd_dim(from, to, ps, :, dim) = xpnd;
                  
                end
              end
            else
              for cl=1:size(obj.expnd_dim,4)
                for ps=1:3
                  if obj.Rblk_size(ps,cl,dim) == ...
                      obj.Rblk_size(pos(dim), clr, dim) &&...
                    obj.Rblk_ofst(ps, cl, dim) ==...
                      obj.Rblk_ofst(pos(dim), clr, dim)

                    obj.expnd_dim(from, to, ps, cl, dim) = xpnd;
                   
                  end
                end
              end
            end
          case 6
            if from >= obj.BLK_STT_EXTND && to >= obj.BLK_STT_EXTND
              obj.expnd_clr(from, to, :, :, :, :) = xpnd;
            elseif from >= obj.BLK_STT_INTRPLT && to >= obj.BLK_STT_INTRPLT
              t = pos(3);
              for h=1:3
                if obj.Rblk_size(h,clr,2) ~= ...
                    obj.Rblk_size(pos(2), clr, 2) ||...
                    obj.Rblk_ofst(h, clr, 2) ~=...
                    obj.Rblk_ofst(pos(2), clr, 2)
                  continue;
                end
                for v= 1:3
                  if obj.Rblk_size(v,clr,1) ~= ...
                      obj.Rblk_size(pos(1), clr, 1) ||...
                      obj.Rblk_ofst(v, clr, 1) ~=...
                      obj.Rblk_ofst(pos(1), clr, 1)
                    continue;
                  end
                  
                  obj.expnd_clr(from, to, v, h, t, :) = xpnd;
                  
                end
              end
            else
              for cl=1:size(obj.expnd_dim,4)
                t = pos(3);
                for h=1:3
                  if obj.Rblk_size(h,cl,2) ~= ...
                      obj.Rblk_size(pos(2), clr, 2) ||...
                      obj.Rblk_ofst(h, cl, 2) ~=...
                      obj.Rblk_ofst(pos(2), clr, 2)
                    continue;
                  end
                  for v= 1:3
                    if obj.Rblk_size(v,cl,1) ~= ...
                        obj.Rblk_size(pos(1), clr, 1) ||...
                        obj.Rblk_ofst(v, cl, 1) ~=...
                        obj.Rblk_ofst(pos(1), clr, 1)
                      continue;
                    end
                    
                    obj.expnd_clr(from, to, v, h, t, cl) = xpnd;
                    
                  end
                end
              end
            end
          otherwise
            if from >= obj.BLK_STT_EXTND && to >= obj.BLK_STT_EXTND
              obj.expnd_blk(from, to, :, :, :) = xpnd;
            elseif from >= obj.BLK_STT_INTRPLT && to >= obj.BLK_STT_INTRPLT
              t = pos(3);
              for h=1:3
                if obj.Rblk_size(h,1,2) ~= ...
                    obj.Rblk_size(pos(2), 1, 2) ||...
                    obj.Rblk_ofst(h, 1, 2) ~=...
                    obj.Rblk_ofst(pos(2), 1, 2)
                  continue;
                end
                for v= 1:3
                  if obj.Rblk_size(v,1,1) ~= ...
                      obj.Rblk_size(pos(1), 1, 1) ||...
                      obj.Rblk_ofst(v, 1, 1) ~=...
                      obj.Rblk_ofst(pos(1), 1, 1)
                    continue;
                  end
                  
                  obj.expnd_blk(from, to, v, h, t, :) = xpnd;
                  
                end
              end
            else
              t = pos(3);
              for h=1:3
                if any(obj.Rblk_size(h,:,2) ~= ...
                    obj.Rblk_size(pos(2),:, 2)) ||...
                    any(obj.Rblk_ofst(h,:, 2) ~=...
                    obj.Rblk_ofst(pos(2),:, 2))
                  continue;
                end
                for v= 1:3
                  if any(obj.Rblk_size(v,:,1) ~= ...
                      obj.Rblk_size(pos(1), :, 1)) ||...
                      any(obj.Rblk_ofst(v, :, 1) ~=...
                      obj.Rblk_ofst(pos(1), :, 1))
                    continue;
                  end
                  
                  obj.expnd_blk(from, to, v, h, t) = xpnd;
                  
                end
              end
            end
        end
      end
    end
    
% Test code - used only when the test code in getExpandMtrx() is used.
%     function xpndr = calcExpandR(~,xpnd, eps)
%       mtx = cell(1,2);
%       [mtx{2}, mtx{1}, ~] = SensingMatrix.invertSVD(xpnd.L, xpnd.U, xpnd.V, eps);
% %       mtx = {xpnd.V.copy(), xpnd.L.copy()};
% %       mtx{2}.invert();
%       xpndr = SensingMatrixCascade.constructCascade(mtx);
%     end
    
    function nxt_blk_indx = calcNxtIndx(obj, blk_indx)
      len_blk_indx = length(blk_indx);
      indx_offset = 3 - len_blk_indx;
      nxbl = blk_indx;
      nxt_blk_indx = [];
      for k=1:len_blk_indx
        kk = k + indx_offset;
        if blk_indx(k) < obj.blk_cnt(kk)
          nxt_blk_indx = nxbl;
          nxt_blk_indx(k) = nxbl(k)+1;
          break;
        else
          nxbl(k)=1;
        end
      end
    end
    
    function pxls = cntOrigRawPxls(obj)
      % cntOrigRawPxls computes the number of original raw pixels in each
      % block in a frame.
      %
      % It takes into account interpolation and block
      % overlap. In case of block overlap each overlapped pixel counts as
      % 1/c, where c is the number of overlapping blocks. Note that for
      % temporal overlap we assume that each pixel can belong to at most two
      % blocks temporally.
      %   Input:
      %     obj - this object
      %   output:
      %     pxls - an array containing the number of pixels per block. The
      %            in the array is determined by calcNxtIndx (which is used
      %            in getBlk().
      
      indx = ones(obj.blk_cnt(1)*obj.blk_cnt(2), 3);
      if isempty(indx)
        pxls = [];
        return
      end
      for k=2:size(indx,1)
        indx(k,:) = obj.calcNxtIndx(indx(k-1,:));
      end
      vrgn = VidRegion(indx, obj);
      clr_pxls = vrgn.n_orig_blk_clr_pxls;
      for iclr = 1:size(obj.ovrlp,1)
        clr_pxls(iclr,:) = clr_pxls(iclr,:) * ...
          (1 - 2*obj.ovrlp(iclr,3)/obj.blk_size(iclr,3));
      end
      pxls = sum(clr_pxls);
    end
    
    function bsz = blkSizes(obj, stt)
      % BlkSizes returns a vector containg the block sizes (of all colosrs)
      % in a blockframe.
      
      bsz = zeros(obj.blk_cnt(1)*obj.blk_cnt(2),1);
      switch stt
        case obj.BLK_STT_RAW
          b=0;
          for v=1:obj.blk_cnt(1)
            for h=1:obj.blk_cnt(2)
              b = b+1;
              indx = [v,h,1];
              [~,~,blk_len, ~,~,~] = obj.blkPosition(indx);
              bsz(b) = sum(prod(blk_len,2));
            end
          end
        case obj.BLK_STT_INTRPLT
          b=0;
          for v=1:obj.blk_cnt(1)
            for h=1:obj.blk_cnt(2)
              b = b+1;
              indx = [v,h,1];
              [~,~,blk_len, ~,~,~] = obj.blkPosition(indx);
              bsz(b) = size(blk_len,1) * prod(blk_len(1,:),2);
            end
          end
        case {obj.BLK_STT_EXTND, obj.BLK_STT_WNDW}
          bsz(:) = sum(prod(obj.blk_size,2));
        case obj.BLK_STT_ZEXT
          bsz(:) =  size(obj.blk_size,1) * obj.ext_Cblk_len;
      end
    end
  end
  
  methods (Static)
    function flds = getBlkInfoFields()
      flds = VidBlocker.blk_info_flds;
    end
    
    function flds = getVidInfoFields()
      flds = VidBlocker.vid_info_flds;
    end
    
    function xpnd = xpndSetNorms(xpnd)
        xpnd.U.setNorm(1, true)
        xpnd.V.setNorm(1, true)
        xpnd.M.setNorm(xpnd.L.getExactNorm(), true);      
    end

  end
  
  methods (Access = protected)
    
    function params = setParamsDefaults(obj, params, raw_vid)
      params.fps = raw_vid.fps;
      if ~isfield(params, 'monochrom')
        params.monochrom = ~raw_vid.UVpresent;
      end
      if ~isfield(params, 'w_type_e')
        params.w_type_e = obj.wnd_enc_type;
      end
      if ~isfield(params, 'w_type_d')
        params.w_type_d = obj.wnd_dec_type;
      end
      if ~isfield(params, 'ovrlp')
        params.ovrlp = [0 0 0];
      end
      if ~isfield(params, 'zero_ext')
        params.zero_ext = obj.zext;
      end
      wrap_ext_defined = ...
        (isstruct(params.zero_ext) && isfield(params.zero_ext,'w')) ||...
        (isnumeric(params.zero_ext) && size(params.zero_ext,1) >= 3);
      if ~isfield(params, 'wrap_ext')
        if ~wrap_ext_defined
          params.wrap_ext = obj.wext;
        end
      elseif wrap_ext_defined
        error('wrap ext defined separately and in zero_ext');
      end
    end
    
    function vec = do_getSingleRawBlk(obj, raw_vid, orig, blk_end)
      blk = cell(size(obj.blk_size,1),1);
      for k=1:size(obj.blk_size,1)
        blk_val = raw_vid{k}(orig(k,1):blk_end(k,1), orig(k,2):blk_end(k,2),...
          orig(k,3):blk_end(k,3));
        blk{k} = blk_val(:);
      end
      vec = vertcat(blk{:});
    end
    
    function setUseGpu(obj,val)
      obj.expnd_dim = obj.setMtrxUseGpu(obj.expnd_dim, val);
      obj.expnd_clr = obj.setMtrxUseGpu(obj.expnd_clr, val);
      obj.expnd_blk = obj.setMtrxUseGpu(obj.expnd_blk, val);
      if ~isempty(obj.vid_info)
        obj.vid_info.use_gpu = val;
      end
    end
    
    function setUseSingle(obj,val)
      obj.expnd_dim = obj.setMtrxUseSingle(obj.expnd_dim, val);
      obj.expnd_clr = obj.setMtrxUseSingle(obj.expnd_clr, val);
      obj.expnd_blk = obj.setMtrxUseSingle(obj.expnd_blk, val);
      if ~isempty(obj.vid_info)
        obj.vid_info.use_single = val;
      end
    end
    
  end
  
  methods (Access=private)
    
    function init(obj, params, raw_vid)
      obj.wnd_enc_type = params.w_type_e;
      obj.wnd_dec_type = params.w_type_d;
      
      if any(2*params.ovrlp > params.yb_size)
        error('overlap %s cannot exceed half the block size %s',...
          show_str(params.ovrlp), show_str(params.yb_size));
      end
      
      obj.fps = params.fps;
      
      % Set zext and wext
      if isstruct(params.zero_ext);
        obj.zext = [params.zero_ext.b; params.zero_ext.f];
        obj.wext = params.zero_ext.w;
      elseif size(params.zero_ext,1) == 3
        obj.zext = params.zero_ext(1:2,:);
        obj.wext = params.zero_ext(3,:);
      else
        obj.zext = params.zero_ext;
        obj.wext  = params.wrap_ext;
      end
      
      % Compute size of raw video in each color
      obj.vid_info = raw_vid;
      yhgt = obj.vid_info.yHeight();
      ywdt = obj.vid_info.yWidth();
      nfr = obj.vid_info.n_frames;
      if ~params.monochrom
        n_clr = 3;
        uvhgt = obj.vid_info.uvHeight();
        uvwdt = obj.vid_info.uvWidth();
        vsize = [yhgt, ywdt, nfr; uvhgt, uvwdt, nfr; uvhgt, uvwdt, nfr];
      else
        n_clr = 1;
        vsize = [yhgt, ywdt, nfr];
      end
      obj.vid_size = vsize;
      
      if size(vsize,1) > 1
        obj.uv_ratio = vsize(1,:) ./ vsize(2,:);
        if (~isfield(params, 'linient') || ~params.linient) &&...
            any(any(mod([params.yb_size; params.ovrlp], ...
            ones(size([params.yb_size;params.ovrlp],1),1)*obj.uv_ratio)));
          error('yb_size and ovrlp must be divisible by uv_ratio');
        end
        uv_size =  floor(params.yb_size ./ obj.uv_ratio);
        uv_ovrlp =  floor(params.ovrlp ./ obj.uv_ratio);
        obj.blk_size = [params.yb_size; uv_size; uv_size];
        obj.ovrlp = [params.ovrlp; uv_ovrlp; uv_ovrlp];
      else
        obj.uv_ratio = [];
        obj.blk_size = params.yb_size;
        obj.ovrlp = params.ovrlp;
      end
      
      % Calculate the number of blocks and color blocks in a video.
      bsize = obj.blk_size(1,:);
      olp = obj.ovrlp(1,:);

      obj.blk_cnt = (obj.vid_size(1,:)-olp)./(bsize-olp);
      obj.blk_cnt([1 2]) = ceil(obj.blk_cnt([1 2]));
      obj.blk_cnt(3) = floor(obj.blk_cnt(3));
      obj.Cblk_cnt = [size(obj.blk_size,1) obj.blk_cnt];
      
%       if any(mod(vsize(2,1:2) - params.ovrlp(1:2), 2))
%         error('For Y component, video_size - overlap must be even');
%       end
      
      % Compute slack, caused by blocks exceeding video in horizontal and
      % vertical dimensions
      slack = (ones(n_clr,1) * obj.blk_cnt) .* (obj.blk_size - obj.ovrlp) +...
        obj.ovrlp - obj.vid_size;
      slack(:,3) = 0; % Not applies to temporal
      if any(mod(slack(:), 2))
        error('slack=%s must be even.', show_str(slack));
      end
      slack = slack/2;
      
      % compute window
      if ~isempty(obj.wnd_enc_type)
        obj.wnd_enc = struct('b',[],'f',[]); % for (V,H,W)
        obj.wnd_enc.b = initWindow(obj.wnd_enc_type);
        for k=1:3
          obj.wnd_enc.f{k} = obj.wnd_enc.b{k}(end:-1:1);
        end
      end
      if ~isempty(obj.wnd_dec_type)
        obj.wnd_dec = struct('b',[],'f',[]); % for (V,H,W)
        obj.wnd_dec.b = initWindow(obj.wnd_dec_type);
        for k=1:3
          obj.wnd_dec.f{k} = 1 - obj.wnd_dec.b{k};
        end
      end
      
      % Set sizes
      obj.ext_Cblk_size = sum([obj.blk_size(1,:);obj.zext;obj.wext]);
      obj.ext_Cblk_len = prod(obj.ext_Cblk_size);
      
      obj.Rblk_size = zeros(3,n_clr,3);
      obj.Rblk_ofst = zeros(3,n_clr,3);
      for k=1:3
        if obj.blk_cnt(k) == 1
          for i=1:3
            obj.Rblk_size(i,:,k) = obj.blk_size(:,k) - 2*slack(:,k);
            if k<3
              obj.Rblk_ofst(i,:,k) = slack(:,k);
            end
          end
        else
          obj.Rblk_size(obj.MID,:,k) = obj.blk_size(:,k);
          if k < 3
            obj.Rblk_size(obj.FST,:,k) = obj.blk_size(:,k) - slack(:,k);
            obj.Rblk_size(obj.LST,:,k) = obj.Rblk_size(obj.FST,:,k);
            obj.Rblk_ofst(obj.FST,:,k) = slack(:,k);
          else
            obj.Rblk_size(obj.FST,:,k) = obj.blk_size(:,k);
            obj.Rblk_size(obj.LST,:,k) = obj.blk_size(:,k) - 2*slack(:,k);
          end
        end
      end
      obj.Iblk_size = reshape(obj.Rblk_size(:,1,:),...
        [size(obj.Rblk_size,1), size(obj.Rblk_size,3)]);
      
      % Allocate space for expnd_dim, expnd_clr and expnd_blk
      obj.expnd_dim = struct(...
        'M', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, n_clr, 3),...
        'U', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, n_clr, 3),...
        'L', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, n_clr, 3),...
        'V', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, n_clr, 3),...
        'I', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, n_clr, 3),...
        'R', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, n_clr, 3));
      obj.expnd_clr = struct(...
        'M', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3, n_clr),...
        'U', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3, n_clr),...
        'L', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3, n_clr),...
        'V', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3, n_clr),...
        'I', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3, n_clr),...
        'R', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3, n_clr));
      obj.expnd_blk = struct(...
        'M', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3),...
        'U', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3),...
        'L', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3),...
        'V', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3),...
        'I', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3),...
        'R', cell(obj.N_BLK_STT, obj.N_BLK_STT, 3, 3, 3));

      
      function wnd = initWindow(type)
        wnd = cell(1,3);
        for d=1:length(wnd)
          N = params.ovrlp(d);
          if ~N
            wnd{d} = [];
            continue;
          end
          
          switch type
            case CS_EncVidParams.WND_NONE;
              wnd{d} = [];
            case CS_EncVidParams.WND_HANNING
              wnd{d} = 0.5 + 0.5*cos((pi/(N+1))*(N:-1:1)');
            case CS_EncVidParams.WND_TRIANGLE;
              wnd{d} = (1:N)' / (N+1);
            case CS_EncVidParams.WND_SQRT_HANNING
              wnd{d} = sqrt(0.5 + 0.5*cos((pi/(N+1))*(N:-1:1)'));
            case CS_EncVidParams.WND_SQRT_TRIANGLE
              wnd{d} = sqrt((1:N)' / (N+1));
            otherwise
              error('Unknown window type');
          end
        end
      end
    end
    
    function blk = interpolate(obj, blk)
      if iscell(blk)
        for k=2:size(obj.blk_size,1)
          blk{k} = obj.interpolate(blk{k});
        end
      else
        for dim=1:3
          uvr = double(obj.uv_ratio(dim));
          if uvr == 1
            continue;
          end
          wgt0 = (uvr:-1:1)/uvr;
          wgt1 = 1 - wgt0;
          sz_in = size(blk);
          sz_slc = sz_in;
          sz_slc(dim)=1;
          blk0 = blk;
          edge_indcs_1 = md_slice_indc(sz_in, dim, sz_in(dim));
          %duplicate end edge;
          blk1 = cat(dim, blk, reshape(blk(edge_indcs_1),sz_slc));
          edge_indcs_0 = md_slice_indc(size(blk1), dim, 1);
          blk1 = blk1(:);
          blk1(edge_indcs_0) = []; % remove first edge
          
          sz_out = sz_in;
          sz_out(dim) = sz_in(dim) * uvr;
          blk = zeros(sz_out);
          blk = blk(:);
          for k = 1:uvr
            indcs = md_slice_indc(sz_out, dim, k:uvr:sz_out(dim));
            blk(indcs) = wgt0(k)*blk0(:) + wgt1(k)*blk1;
          end
          blk = reshape(blk, sz_out);
        end
      end
    end
    
    function blk = undoInterpolate(obj, blk)
      if iscell(blk)
        for k=2:size(obj.blk_size,1)
          blk{k} = obj.undoInterpolate(blk{k});
        end
      else
        blk = blk(:,:,(1:obj.uv_ratio(3):size(blk,3)));
        blk = blk(:,(1:obj.uv_ratio(2):size(blk,2)),:);
        blk = blk((1:obj.uv_ratio(1):size(blk,1)),:,:);
      end
    end
    
    function blk_ext = extendBlk(obj, blk, ofst)
      if iscell(blk)
        blk_ext = cell(size(blk));
        for k=1:size(obj.blk_size,1)
          blk_ext{k} = obj.extendBlk(blk{k},ofst);
        end
      else
        bsz = obj.blk_size(1,:);
        blk_ext = obj.zeros(bsz);
        sz = ones(1,3);
        sz1 = size(blk);
        sz(1:length(sz1)) = sz1;
        bbgn = ofst + 1;
        bend = ofst + sz;
        blk_ext(bbgn(1):bend(1), bbgn(2):bend(2), bbgn(3):bend(3)) = ...
          obj.toFloat(blk);
        
        % Extend V
        for j=1:bbgn(1)-1
          blk_ext(j,bbgn(2):bend(2), bbgn(3):bend(3)) = ...
            blk_ext(bbgn(1),bbgn(2):bend(2), bbgn(3):bend(3));
        end
        for j = bend(1)+1:bsz(1)
          blk_ext(j,bbgn(2):bend(2), bbgn(3):bend(3)) = ...
            blk_ext(bend(1),bbgn(2):bend(2), bbgn(3):bend(3));
        end
        
        % extend H
        for j=1:bbgn(2)-1
          blk_ext(:,j, bbgn(3):bend(3)) = ...
            blk_ext(:,bbgn(2), bbgn(3):bend(3));
        end
        for j = bend(2)+1:bsz(2)
          blk_ext(:,j, bbgn(3):bend(3)) = ...
            blk_ext(:,bend(2), bbgn(3):bend(3));
        end
        
        % extend T
        for j=1:bbgn(3)-1
          blk_ext(:,:,j) = blk_ext(:,:,bbgn(3));
        end
        for j = bend(3)+1:bsz(3)
          blk_ext(:,:,j) = blk_ext(:,:,bend(3));
        end
      end
    end
    
    function blk = undoExtendBlk(obj, blk, blk_ofst, blk_len)
      if iscell(blk)
        for k=1:size(obj.blk_size,1)
          blk{k} = obj.undoExtendBlk(blk{k}, blk_ofst, blk_len);
        end
      else
        bbgn = blk_ofst+1;
        bend = blk_ofst + blk_len;
        blk = blk(bbgn(1):bend(1),bbgn(2):bend(2),bbgn(3):bend(3));
      end
    end
    
    function xpnd = do_getExpandMtrx(obj, from, to, blk_indx, eps, clr, dim)
      if nargin < 5
        eps = 0;
      end
      if nargin < 7
        xpnd = struct('M',[],'U',[],'L',[],'V',[], 'I',[],'R',[]);
        if nargin < 6
          n_clr = size(obj.blk_size,1); % number of colors
          args = {from, to, blk_indx, eps};
          xp  = struct(...
            'M', cell(1, n_clr), 'U', cell(1, n_clr),...
            'L', cell(1, n_clr),'V', cell(1, n_clr),...
            'I', cell(1, n_clr), 'R', cell(1, n_clr));
          for iclr = 1:n_clr
            args_k = [args {iclr}];
            xp(iclr) = obj.do_getExpandMtrx(args_k{:});
          end
          
          xpnd.M = SensingMatrixBlkDiag.constructBlkDiag({xp(:).M}');
          xpnd.M.use_gpu = obj.use_gpu;
          xpnd.M.use_single = obj.use_single;
          [xpnd.L, xpnd.U, xpnd.V] = SensingMatrixBlkDiag.compSVDfromBlks({...
            xp(:).L}', {xp(:).U}', {xp(:).V}', eps);
          xpnd = obj.setMtrxUseGpu(xpnd, obj.use_gpu);
          xpnd = obj.setMtrxUseSingle(xpnd, obj.use_single);
          xpnd = obj.xpndSetNorms(xpnd);
        else
          pv = obj.cmpPos(blk_indx,clr,1);
          ph = obj.cmpPos(blk_indx,clr,2);
          pt = obj.cmpPos(blk_indx,clr,3);
          if ~isempty(obj.expnd_clr(from,to,pv,ph,pt,clr).M)
            xpnd = obj.expnd_clr(from,to,pv,ph,pt,clr);
          else
            n_case = 3; % number of dimensions
            args = {from, to, blk_indx, eps, clr};
            xp  = struct(...
              'M', cell(n_case, 1), 'U', cell(n_case, 1),...
              'L', cell(n_case, 1), 'V', cell(n_case, 1),...
              'I', cell(n_case, 1), 'R', cell(n_case, 1));
            for k=1:n_case
              args_k = [args {k}];
              xp(k) = obj.do_getExpandMtrx(args_k{:});
            end
            
            xpnd.M = SensingMatrixKron.constructKron({xp(end:-1:1).M}');
            xpnd.M.use_gpu = obj.use_gpu;
            xpnd.M.use_single = obj.use_single;
            [xpnd.L, xpnd.U, xpnd.V] = SensingMatrixKron.compSVDfromTerms({...
              xp(end:-1:1).L}', {xp(end:-1:1).U}', {xp(end:-1:1).V}', eps);
            xpnd = obj.setMtrxUseGpu(xpnd, obj.use_gpu);
            xpnd = obj.setMtrxUseSingle(xpnd, obj.use_single);
            xpnd = obj.xpndSetNorms(xpnd);
            % Copy to neighbors
            if from >= obj.BLK_STT_EXTND && to >= obj.BLK_STT_EXTND
              obj.expnd_clr(from, to, :, :, :, :) = xpnd;
            elseif from >= obj.BLK_STT_INTRPLT && to >= obj.BLK_STT_INTRPLT
              obj.expnd_clr(from, to, pv, ph, pt, :) = xpnd;
            elseif clr>1
              obj.expnd_clr(from, to, pv, ph, pt, 2:end) = xpnd;
            else
              obj.expnd_clr(from, to, pv, ph, pt, clr) = xpnd;
            end
            
          end
          
        end
        
        return
      end
            
      pos = obj.cmpPos(blk_indx,clr,dim);
      
      do_copy = false;
      if isempty(obj.expnd_dim(from, to, pos, clr, dim).M)
        xpnd = obj.calcExpandMtrx(from, to, pos, eps, clr, dim);
        xpnd = obj.setMtrxUseGpu(xpnd, obj.use_gpu);
        xpnd = obj.setMtrxUseSingle(xpnd, obj.use_single);
        do_copy = true;
      else
        xpnd = obj.expnd_dim(from, to, pos, clr, dim);
      end
      if isempty(xpnd.U)
         [xpnd.L,xpnd.U,xpnd.V] = xpnd.M.compSVD(eps,  false);
        do_copy = true;
      end
      
     if do_copy
       obj.copy_to_ngbrs(from, to, pos, clr, dim, xpnd);
     end
      
      xpnd = obj.expnd_dim(from, to, pos, clr, dim);
    end
    
    function ps = cmpPos(obj, b_idx, clr, dm)
      if nargin == 3
        ps = [0,0,0];
        for k=1:3
          ps(k) = obj.cmpPos(b_idx,clr,k);
        end
      elseif isscalar(b_idx)
        ps = b_idx;
      elseif b_idx(dm) > 1 && b_idx(dm) < obj.blk_cnt(dm)
        ps = 2;
      elseif b_idx(dm) == 1 && (...
          (isempty(clr) && any(obj.Rblk_ofst(obj.FST,:,dm))) ||...
          (~isempty(clr) && obj.Rblk_ofst(obj.FST,clr,dm)));
        ps = 1;
      elseif b_idx(dm) == obj.blk_cnt(dm) && (...
          (isempty(clr) && ...
          any(obj.Rblk_size(obj.LST,:,dm)' < obj.blk_size(:,dm))) ||...
          (~isempty(clr) && ...
          obj.Rblk_size(obj.LST,clr,dm) < obj.blk_size(clr,dm)))
        ps = 3;
      else
        ps = 2;
      end
    end
    
    function xpnd = calcExpandMtrx(obj, from, to, pos, eps, clr, dim)
      xpnd = struct('M',[],'U',[],'L',[],'V',[], 'I',[],'R',[]);
      if from == to
        switch from
          case obj.BLK_STT_RAW
            xpnd = build_unit(obj.Rblk_size(pos,clr,dim));
          case obj.BLK_STT_INTRPLT
            xpnd = build_unit(obj.Iblk_size(pos,dim));
          case {obj.BLK_STT_EXTND, obj.BLK_STT_WNDW}
            xpnd = build_unit(obj.blk_size(1,dim));
          case obj.BLK_STT_ZEXT;
            xpnd = build_unit(obj.ext_Cblk_size(1,dim));
        end
      elseif from > to
        invx = obj.do_getExpandMtrx(to, from, pos, eps, clr, dim);
        xpnd = struct('M',[], 'U',[], 'L',[], 'V', [], 'I',[],'R',[]);
        [xpnd.L, xpnd.U, xpnd.V] = ...
          SensingMatrix.invertSVD(invx.L, invx.U, invx.V, eps);
        xpnd.M = SensingMatrix.constructSVD(xpnd.L, xpnd.U, xpnd.V);
        
        SensingMatrix.chkSVD(xpnd.L, xpnd.U, xpnd.V, 2*eps, xpnd.M, true);
      elseif to - from > 1
        if isempty(obj.expnd_dim(to-1, to, pos, clr, dim).M);
          c1 = obj.calcExpandMtrx(to-1, to, pos, eps, clr, dim);
          obj.copy_to_ngbrs(to-1, to, pos, clr, dim, c1);
        else
          c1 = obj.expnd_dim(to-1, to, pos, clr, dim);
        end
        if isempty(obj.expnd_dim(from, to-1, pos, clr, dim).M);
          c2 = obj.calcExpandMtrx(from, to-1, pos, eps, clr, dim);
          obj.copy_to_ngbrs(from, to-1, pos, clr, dim, c2);
        else
          c2 = obj.expnd_dim(from, to-1, pos, clr, dim);
        end
        xpnd.M = SensingMatrixCascade.constructCascade({c1.M, c2.M});
      elseif from == obj.BLK_STT_RAW
        if clr==1 || obj.uv_ratio(dim) == 1
          xpnd = build_unit(obj.Rblk_size(pos, clr, dim));
        else
          % Compute interpolation matrix
          sz_in = obj.Rblk_size(pos, clr, dim);
          sz_out = obj.Iblk_size(pos, dim);
%           rr=zeros(sz_out,2);
%           cc=zeros(sz_out,2);
%           vv=zeros(sz_out,2);
%           uvr = double(obj.uv_ratio(dim));
%           for k=1:uvr
%             rr((k:uvr:sz_out),1) = (k:uvr:sz_out)';
%             rr((k:uvr:sz_out),2) = (k:uvr:sz_out)';
%             cc((k:uvr:sz_out),1) = (1:sz_in)';
%             cc((k:uvr:sz_out),2) = [(2:sz_in)';sz_in];
%             vv((k:uvr:sz_out),1) = (uvr+1-k)/uvr;
%             vv((k:uvr:sz_out),2) = (k-1)/uvr;
%           end
%           mt = sparse(rr(:),cc(:),vv(:));
%           xpnd.M = SensingMatrixMatlab(mt);
          xpnd.M = construct_intrplt_mtx(sz_in, sz_out);
        end
      elseif from == obj.BLK_STT_INTRPLT
        % Extending the block
        sz_out = obj.blk_size(1,dim);
        sz_in = obj.Iblk_size(pos,dim);
        if sz_in == sz_out
          xpnd = build_unit(sz_in);
        else
%           rr = (1:sz_out);
%           cc = [ones(1,b_ofst), (1:sz_in), sz_in*ones(1,e_ofst)];
%           xpnd.M = SensingMatrixMatlab(sparse(rr,cc,1));
%           xpnd.M.setPsdOrthoCol(true);
          
          if sz_in == 1
            xpnd.M = SensingMatrixDC.construct(sz_out, true, false);
          else
            b_ofst = obj.Rblk_ofst(pos,1,dim);
            e_ofst = sz_out - b_ofst - sz_in;
            mtx = cell(1,3);
            if b_ofst > 0
              n_mtx = 1;
              mtx{1} = SensingMatrixDC.construct(b_ofst+1, true, false);
              sz_in = sz_in-1;
            else
              n_mtx = 0;
            end
            if e_ofst > 0
              sz_in = sz_in-1;
              if sz_in > 0
                n_mtx = n_mtx+1;
                mtx{n_mtx} = SensingMatrixUnit(sz_in);
              end
              n_mtx = n_mtx+1;
              mtx{n_mtx} = SensingMatrixDC.construct(e_ofst+1,true,false);
            else
              n_mtx = n_mtx+1;
              mtx{n_mtx} = SensingMatrixUnit(sz_in);
            end
            xpnd.M = SensingMatrixBlkDiag.construct(mtx(1:n_mtx));
          end
            
%           if ~isequal(sparse(mt.getMatrix()), xpnd.M.getMatrix())
%             error('matrices do not match');
%           end
        end
      elseif from == obj.BLK_STT_EXTND
        wgt_b = obj.wnd_enc.b{dim};
        wgt_f = obj.wnd_enc.f{dim};
        dg = ones(obj.blk_size(1,dim),1);
        dg(1:length(wgt_b)) = wgt_b;
        dg(end-length(wgt_f)+1:end) = wgt_f;
        xpnd.M = SensingMatrixDiag.constructDiag(dg);
      elseif from == obj.BLK_STT_WNDW
        sz_out = obj.ext_Cblk_size(dim);
        sz_in = obj.blk_size(1,dim);
        if sz_in == sz_out
          xpnd = build_unit(sz_in);
        elseif obj.wext(dim) == 0
          ofst = obj.zext(1,dim);
          xpnd.M = SensingMatrixSelectRange.constructSelectRange(...
            ofst+1, ofst+sz_in, sz_out);
          xpnd.M.transpose();
        else
          dff = sz_out-sz_in;
          rr = [(1:sz_out),(sz_in+1:sz_out)];
          cc = [(1:sz_in), ones(1,dff), sz_in*ones(1,dff)];
          vv = [ones(1,sz_in), [(1:dff), (dff:-1:1)]/(dff+1)];
          xpnd.M = SensingMatrixMatlab(sparse(rr,cc,vv));
          
          mt = SensingMatrixConcat({SensingMatrixUnit(sz_in),...
            SensingMatrixCascade.construct({...
            SensingMatrixMatlab([1:dff;dff:-1:1]'/(dff+1)),...
            SensingMatrixSelect.construct([1,sz_in],sz_in),})...            
            });

          if ~isequal(sparse(mt.getMatrix()), xpnd.M.getMatrix())
            error('matrices do not match');
          end
        end
      end
      
      function xp = build_unit(lng)
        xp = struct('M', SensingMatrixUnit(lng), ...
          'U', SensingMatrixUnit(lng),...
          'L', SensingMatrixUnit(lng),...
          'V', SensingMatrixUnit(lng),...
          'I', SensingMatrixUnit(lng),...
          'R', SensingMatrixUnit(lng));
      end
      
      xpnd = obj.setMtrxUseGpu(xpnd, obj.use_gpu);
      xpnd = obj.setMtrxUseSingle(xpnd, obj.use_single);

    end
   
    function copy_to_ngbrs(obj, from, to, pos, clr, dim, xpnd)
      if ~isempty(xpnd.U)
        xpnd = obj.xpndSetNorms(xpnd);
      end
      
      if from >= obj.BLK_STT_EXTND && to >= obj.BLK_STT_EXTND
        obj.expnd_dim(from,to,:,:,dim) = xpnd;
      elseif from >= obj.BLK_STT_INTRPLT && to >= obj.BLK_STT_INTRPLT
         obj.expnd_dim(from,to,pos,:,dim) = xpnd;
      elseif clr>1
        obj.expnd_dim(from,to,pos,2:end,dim) = xpnd;
      else
        obj.expnd_dim(from, to, pos, clr, dim) = xpnd;
      end
    end
        
  end
  
  methods (Access=protected, Static)
    
    function params = compParamsFromOpts(params, raw_vid, opts)
      params.yb_size = opts.blk_size;
      params.ovrlp =  opts.blk_ovrlp;
      if opts.blk_dur > 0
        nfr = max(1,ceil(opts.blk_dur * raw_vid.fps));
        params.ovrlp(3) = min(nfr-1, ...
          round(nfr * params.ovrlp(3)/params.yb_size(3)));
        params.yb_size(3) = nfr;
      end
      if opts.sav_levels >= 0
        m = pow2(opts.sav_levels);
        params.yb_size(1:2) = [raw_vid.yHeight(), raw_vid.yWidth()]/m;
        params.ovrlp(1:2) = params.yb_size(1:2)/2;
      end
      params.w_type_e = opts.wnd_type;
      params.monochrom = ~opts.process_color || ~raw_vid.UVpresent;
      params.zero_ext = [opts.zero_ext_b; opts.zero_ext_f];
      params.wrap_ext = opts.wrap_ext;
    end
    
    function blk = windowBlk(blk, wnd, is_first, is_last)
      if iscell(blk)
        for k=1:length(blk);
          blk{k} = VidBlocker.windowBlk(blk{k}, wnd, is_first, is_last);
        end
      else
        
        sz = size(blk);
        
        if isempty(wnd)
          return
        end
        
        for k=1:3
          if isempty(wnd.b{k})
            continue;
          end
          if ~is_first(k)
            wgt_b = wnd.b{k};
            for i=1:length(wgt_b)
              slc_b = md_slice_indc(sz, k, i);
              blk(slc_b) = blk(slc_b) .* wgt_b(i);
            end
          end
          if ~is_last(k)
            wgt_f = wnd.f{k};
            l_wgt = length(wgt_f);
            for i=1:l_wgt
              slc_f = md_slice_indc(size(blk), k, sz(k)-l_wgt+i);
              blk(slc_f) = blk(slc_f) .* wgt_f(i);
            end
          end
        end
      end
    end
    
    function blk = unWindowBlk(blk, wnd, is_first, is_last)
      if iscell(blk)
        for k=1:length(blk);
          blk{k} = VidBlocker.unWindowBlk(blk{k}, wnd, is_first, is_last);
        end
      else
        
        sz = size(blk);
        
        if isempty(wnd)
          return
        end
        
        for k=1:3
          if isempty(wnd.b{k})
            continue;
          end
          if ~is_first(k)
            wgt_b = wnd.b{k};
            for i=1:length(wgt_b)
              slc_b = md_slice_indc(sz, k, i);
              blk(slc_b) = blk(slc_b) ./wgt_b(i);
            end
          end
          if ~is_last(k)
            wgt_f = wnd.f{k};
            l_wgt = length(wgt_f);
            for i=1:l_wgt
              slc_f = md_slice_indc(size(blk), k, sz(k)-l_wgt+i);
              blk(slc_f) = blk(slc_f) ./ wgt_f(i);
            end
          end
        end
      end
    end
    
    function mtxs = setMtrxUseGpu(mtxs, val)
      if iscell(mtxs)
        for k=1:numel(mtxs)
          if isempty(mtxs{k})
            continue;
          end
          mtxs{k} = VidBlocker.setMtrxUseGpu(mtxs{k}, val);
        end
      elseif isstruct(mtxs)
        if isempty(mtxs)
          return
        end
        flds = fieldnames(mtxs);
        for kf=1:numel(flds)
          fld = flds{kf};
          for k=1:numel(mtxs)
            if isempty(mtxs(k).(fld))
              continue;
            end
            mtxs(k).(fld) = VidBlocker.setMtrxUseGpu(mtxs(k).(fld), val);
          end
        end
      elseif isa(mtxs, 'SensingMatrix')
        mtxs.use_gpu = val;
      end
    end
    
    function mtxs = setMtrxUseSingle(mtxs, val)
      if iscell(mtxs)
        for k=1:numel(mtxs)
          if isempty(mtxs{k})
            continue;
          end
          mtxs{k} = VidBlocker.setMtrxUseSingle(mtxs{k}, val);
        end
      elseif isstruct(mtxs)
        if isempty(mtxs)
          return
        end
        flds = fieldnames(mtxs);
        for kf=1:numel(flds)
          fld = flds{kf};
          for k=1:numel(mtxs)
            if isempty(mtxs(k).(fld))
              continue;
            end
            mtxs(k).(fld) = VidBlocker.setMtrxUseSingle(mtxs(k).(fld), val);
          end
        end
      elseif isa(mtxs, 'SensingMatrix')
        mtxs.use_single = val;
      end
    end
  end
  
end

