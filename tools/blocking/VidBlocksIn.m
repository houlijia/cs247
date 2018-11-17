classdef VidBlocksIn < VidBlocker
    %VidBlocksIn Enables reading blocks from a file, while keeping
    %   a buffer of only necessary frames
    
    properties
        % frames available in buffer
        frms_cnt=0;
        frms_offset=0;
        default_stt;  %Default state of read in blocks (BLK_STT_...)
        
        % Video data (cell array) of frms_cnt frames, beginning at frame
        % skip_frms+frms_offset+1
%         vid_data;
    end
    
    properties (Access = private)
      % handle for mex operations
      mex_handle = [];
      
      % Allowed range in file
      skip_frms=0;
      
      close_on_exit = false;
      
      % For use in getFrmBlks
      b_bgn_v;
      b_bgn_h;
      b_end_v;
      b_end_h;
      b_frm_xpnd_mtrx= [];
      b_n_mlti_frm = 0;
      b_mlti_frm_xpnd_mtrx = [];
      
      % Offsets to the beginning of each color block in a vector of raw data
      % for a whole frame
      frm_cblk_vec_ofsts;
      % Length of a color block in a vector of raw data
      frm_cblk_len;
      % Offsets to the beginning of each block in a vector of raw data for a
      % whole frame
      frm_blk_vec_ofsts;
      % Length of a  block in a vector of raw data
      frm_blk_len;
      % The total length of a raw vecotr of blocks of a whole frame
      frm_blk_vec_len;
      
      % The maximal length of a color block
      max_cblk_len;
    end
    
    methods
      function obj = VidBlocksIn(params, src, def_stt, opts, intrplt)
        % Constructor
        %   params - A params struct suitalbe for the constructor
        %            of a VidBlocker, or a VidBlocker object as
        %            an example.  the field monochrom will be set if
        %            the file is such.
        %   src - A path to a file containing a JSON descriptor of the
        %         input or a RawVidInfo object. If params is a VidBlocker,
        %         params.vid_info must be of the same format as src.
        %   def_stt - Default state of read in blocks (BLK_STT_...)
        %   opts - (optional) if not empty, can be either a CS_EncVidParams
        %          objects, used to construct VidBlocker, or a struct with
        %          the optional fields:
        %             nfrm - maximal number of frames to read. Default,or
        %                     if nfrm==-1: no. of frames in the file.
        %             skip - Number of frames to skip from the
        %                    beginning of the file. Default=0.
        %          If opts is a CS_EncVidParams, these values are taken from
        %          it.
        %   intrplt: (optional) If true, whenever a frame is read, the
        %          UV components(if present) are interplolated to the same
        %          size as the Y component. Otherwise, they are left at the
        %          same size.If params.intrplt>1, all components are
        %          interpolated to a size such that in each dimension it is a
        %          multiple of params.intrplt. Ignored if src is RawvidInfo.
        %          Default: 0, or determined from opts or params.
        if nargin < 5
          if isa(params, 'VidBlocker') && ~isempty(params.vid_info.intrplt_mtx)
            intrplt = params.vid_info.intrplt_mtx.lvl;
          else
            intrplt = 0;
          end
        end
        
        if nargin < 4 || isempty(opts)
          args = {params, []};
          skip = 0;
          nfrm = inf;
        elseif isa(opts, 'CS_EncVidParams')
          if isa(params, 'VidBlocker')
            error('opts should not be a CS_EncVidParams when params is VidBlocker');
          end
          args = {params, [], opts};
          nfrm = opts.n_frames;
          skip = opts.start_frame - 1;
          if nargin < 5 && opts.sav_levels >= 0
            intrplt = pow2(opts.sav_levels);
          end
        else
          args = {params, []};
          if isfield(opts, 'nfrm')
            nfrm = opts.nfrm;
          else
            nfrm = inf;
          end
          if isfield(opts,'skip')
            skip = opts.skip;
          else
            skip = 0;
          end
        end
        if nfrm == -1
          nfrm = inf;
        end
        
        if ~isa(src, 'RawVidInfo')
          [f_info, ~, err_msg] = read_raw_video(src, 0, skip+1,...
            struct('intrplt', intrplt));
          if ~isempty(err_msg)
            error('failed to read %s (%s)', src, err_msg);
          end
          src = f_info;
          clex = true;
        else
          clex = false;
        end
        
        if src.n_frames > nfrm+skip
          src.n_frames = nfrm+skip;
        end
        
        args{2} = src;
        
        if isa(params, 'VidBlocker')
          if ~src.isEqual(params.vid_info)
            error('src not equal to params.vid_info');
          end
          args{1} = params.getParams();
        end
        
        obj = obj@VidBlocker(args{:});
        
        obj.default_stt = def_stt;
        
        if ~all(obj.blk_size(:))
          error('zero entries in blk_size');
        end
        obj.skip_frms = skip;
        obj.close_on_exit = clex;
        
        % Read one frame and create mex_handle
%         [~, obj.vid_data, err_msg] = read_raw_video(obj.vid_info, 1, obj.skip_frms + 1);
        [~, vid_data, err_msg] = read_raw_video(obj.vid_info, 1, obj.skip_frms + 1);
        if ~isempty(err_msg)
          error('reading input forwards failed (%s)', err_msg);
        end
        obj.frms_cnt  = 1;
%         obj.mex_handle = initRawVidBlocker_mex(obj.vid_data, ...
%           uint16(obj.blk_size), uint16(obj.ovrlp));
        obj.mex_handle = initRawVidBlocker_mex(vid_data, ...
          uint16(obj.blk_size), uint16(obj.ovrlp));
      end
      
      function delete(obj)
        if ~isempty(obj.mex_handle)
          deleteRawVidBlocker_mex(obj.mex_handle);
        end
        if obj.close_on_exit
          delete(obj.vid_info);
          obj.vid_info = [];
        end
      end
      
      function setBlkFrm(obj, def_n_frms)
        % Sets parameters for getting whole block frames at once.
        % If the arguments have already been set, the function returns without
        % doing anything:
        %   Input:
        %     obj - this object
        %     def_n_frms - optional specifier for the default number of frame 
        %     blocks which are to be read at once.
        
        if isempty(obj.b_frm_xpnd_mtrx)
          % Compute b_bgn_v, b_bgn_h, b_end_v, b_end_h
          n_clr = size(obj.blk_size,1);
          obj.b_end_v = zeros(n_clr, obj.blk_cnt(1));
          obj.b_end_h = zeros(n_clr, obj.blk_cnt(2));
          b_sz = obj.blk_size - obj.ovrlp;
          
          obj.b_bgn_v = b_sz(:,1) * (0:obj.blk_cnt(1)-1) + 1;
          obj.b_bgn_v(:,2:end) = obj.b_bgn_v(:,2:end) - ...
            reshape(obj.Rblk_ofst(obj.FST,:,1),n_clr,1)*...
            ones(1,obj.blk_cnt(1)-1);
          
          obj.b_bgn_h = b_sz(:,2) * (0:obj.blk_cnt(2)-1) + 1;
          obj.b_bgn_h(:,2:end) = obj.b_bgn_h(:,2:end) - ...
            reshape(obj.Rblk_ofst(obj.FST,:,2),n_clr,1)*...
            ones(1,obj.blk_cnt(2)-1);
          
          obj.b_end_v(:,1) = obj.b_bgn_v(:,1) + ...
            reshape(obj.Rblk_size(obj.FST,:,1), n_clr,1);
          obj.b_end_v(:,end) = obj.b_bgn_v(:,end) + ...
            reshape(obj.Rblk_size(obj.LST,:,1), n_clr,1);
          if obj.blk_cnt(1) > 2
            obj.b_end_v(:,2:end-1) = obj.b_bgn_v(:,2:end-1) + ...
              reshape(obj.Rblk_size(obj.MID,:,1), n_clr,1) *...
              ones(1, obj.blk_cnt(1)-2);
          end
          obj.b_end_v = obj.b_end_v - 1;
          
          obj.b_end_h(:,1) = obj.b_bgn_h(:,1) + ...
            reshape(obj.Rblk_size(obj.FST,:,2), n_clr,1);
          obj.b_end_h(:,end) = obj.b_bgn_h(:,end) + ...
            reshape(obj.Rblk_size(obj.LST,:,2), n_clr,1);
          if obj.blk_cnt(1) > 2
            obj.b_end_h(:,2:end-1) = obj.b_bgn_h(:,2:end-1) + ...
              reshape(obj.Rblk_size(obj.MID,:,2), n_clr,1) * ...
              ones(1, obj.blk_cnt(2)-2);
          end
          obj.b_end_h = obj.b_end_h -1;
          
          b_len = zeros(obj.Cblk_cnt(1,1:3));
          b_len_v = obj.b_end_v - obj.b_bgn_v + 1;
          b_len_h = obj.b_end_h - obj.b_bgn_h + 1;
          for c=1:size(b_len,1)
            for v=1:size(b_len,2)
              for h=1:size(b_len,3)
                b_len(c,v,h) =...
                  b_len_v(c,v) * b_len_h(c,h) * obj.Rblk_size(2,c,3);
              end
            end
          end
          obj.frm_cblk_len = b_len(:);
          obj.max_cblk_len = max(b_len(:));
          b_len = reshape(b_len, n_clr, numel(b_len)/n_clr);
          obj.frm_blk_len = sum(b_len);
          obj.frm_blk_len = obj.frm_blk_len(:);
          o_len = cumsum(b_len(:));
          obj.frm_cblk_vec_ofsts = [0; o_len(1:end-1)];
          obj.frm_blk_vec_ofsts = obj.frm_cblk_vec_ofsts(1:n_clr:length(o_len));
          obj.frm_blk_vec_len = o_len(end);
          
          obj.b_frm_xpnd_mtrx = true;
          
          % Create frm_xpnd_mtrx
          stt_mtrx = cell(1, obj.default_stt-obj.BLK_STT_RAW+1);
          mtrx = cell(obj.blk_cnt(1), obj.blk_cnt(2));
          for stt=obj.BLK_STT_RAW:obj.default_stt
            for h=1:obj.blk_cnt(2);
              for v = 1:obj.blk_cnt(1);
                indx = [v,h,1];
                if stt==obj.BLK_STT_RAW
                  xpnd = obj.getExpandMtrx(stt,stt,indx,0);
                else
                  xpnd = obj.getExpandMtrx(stt-1,stt,indx,0);
                end
                mtrx{v,h} = xpnd.M;
              end
            end
            stt_mtrx{end+1-stt} = SensingMatrixBlkDiag.construct(mtrx(:));
          end
           obj.b_frm_xpnd_mtrx = SensingMatrixCascade.construct(stt_mtrx(:));
        end
        
        if nargin >= 2 && def_n_frms && def_n_frms ~= obj.b_n_mlti_frm
          obj.b_mlti_frm_xpnd_mtrx = SensingMatrixKron.construct({...
            SensingMatrixUnit(def_n_frms), obj.b_frm_xpnd_mtrx});
          obj.b_n_mlti_frm = def_n_frms;
        end
          
      end
      
      function blk = getSingleBlk(obj, ~, blk_indx, stt)
        [orig, blk_end, ~, ~, ~, ~] = obj.blkPosition(blk_indx);
        
        if obj.frms_cnt > 0
          n_bck = obj.frms_offset - orig(1,3) + 1;
          n_fwd = blk_end(1,3) - (obj.frms_offset+obj.frms_cnt);
        else
          obj.frms_offset = orig(1,3)-1;
          n_bck = 0;
          n_fwd = blk_end(1,3);
        end
        
        if n_bck > 0
          obj.readBck(n_bck);
        end
        if n_fwd > 0
          obj.readFwd(n_fwd);
        end
        
        % Get raw vector
        vec = getRawVidBlocker_mex(obj.mex_handle, uint32(blk_indx));
        
        % Test code
%         orig(:,3) = orig(:,3) - obj.frms_offset;
%         blk_end(:,3) = blk_end(:,3) - obj.frms_offset;
%         ref_vec = do_getSingleRawBlk(obj, obj.vid_data, orig, blk_end);
%         if ~isequal(ref_vec, vec)
%           error('ref_vec and vec not equal');
%         end
        
        % expand to state stt
        xpnd = obj.getExpandMtrx(obj.BLK_STT_RAW, stt, blk_indx, 0);
        vec = xpnd.M.toFloat(vec);
        blk = xpnd.M.multVec(vec);
      end
      
      function [blk, nxt_blk_indx] = getBlks(obj, blk_indx)
        [blk, nxt_blk_indx] = obj.getBlk([], blk_indx, obj.default_stt);
      end
      
      function [vec, nxt_frm, n_frmblk_read] = getFrmBlks(obj, frst, last)
        % getFrmBlks read the blocks corresponding to several frames,
        % converts them to the right format and returns them as a vector.
        % Blocks are organized first by color, then by vertical, horizontal
        % and temporal numbers.
        % Input:
        %   obj - this object
        %   frst - temporal block index of first frame
        %   last - temporal block index of last frame
        % Output:
        %   vec - output vector
        %   nxt_frm - if 'last' is not the last block index, next_frm is
        %            last+1, otherwise it is empty.
        %   n_frmblk_read - actual number of frame blocks read. Normally,
        %       it is last-frst+1, but at the end of the file it may be less.
        
        if frst<1 || frst > obj.blk_cnt(3) || frst > last
          error('illegal frm block numbers');
        end
        if last > obj.blk_cnt(3)
          last = obj.blk_cnt(3);
        end
        n_frmblk_read = last - frst + 1;
        
        if obj.b_n_mlti_frm == n_frmblk_read
          xpnd_mtx = obj.b_mlti_frm_xpnd_mtrx;
        else
          if isempty(obj.b_frm_xpnd_mtrx)
            obj.setBlkFrm();
          end
          xpnd_mtx = SensingMatrixKron.construct({...
            SensingMatrixUnit(n_frmblk_read), obj.b_frm_xpnd_mtrx});
        end
        
        [orig, ~, ~, ~, ~, ~] = obj.blkPosition([1,1,frst]);
        [~, frms_end, ~, ~, ~, ~] = obj.blkPosition([1,1,last]);
        if obj.frms_cnt > 0
          n_bck = obj.frms_offset - orig(1,3) + 1;
          n_fwd = frms_end(1,3) - (obj.frms_offset+obj.frms_cnt);
        else
          obj.frms_offset = orig(1,3)-1;
          n_bck = 0;
          n_fwd = frms_end(1,3);
        end
        
        if n_bck > 0
          obj.readBck(n_bck);
        end
        if n_fwd > 0
          obj.readFwd(n_fwd);
        end
                
        vec = getRawVidBlocker_mex(obj.mex_handle, uint32(frst), ...
          uint32(n_frmblk_read));
        
        % test code
%         t_stp = obj.blk_size(:,3) - obj.ovrlp(:,3);
%         
%         t_bgn = t_stp*(frst-1) - obj.frms_offset + 1;
%         t_end = t_bgn + obj.blk_size(:,3)-1;
%         ref_vec = zeros(obj.frm_blk_vec_len*n_frmblk_read,1, 'like', obj.vid_data{1});
%         n_clr = size(obj.blk_size,1);
%         for t=frst:last
%           for h=1:obj.blk_cnt(2)
%             for v=1:obj.blk_cnt(1)
%               for c = 1:n_clr
%                 indx = c + n_clr*((v-1) + (h-1)*obj.blk_cnt(1));
%                 ofst = obj.frm_cblk_vec_ofsts(indx) + (t-frst)*obj.frm_blk_vec_len;
%                 blk = ...
%                   obj.vid_data{c}(...
%                   obj.b_bgn_v(c,v):obj.b_end_v(c,v),...
%                   obj.b_bgn_h(c,h):obj.b_end_h(c,h),...
%                   t_bgn(c):t_end(c));
%                 ref_vec(ofst+1:ofst+obj.frm_cblk_len(indx)) = blk(:);
%               end
%             end
%           end
%           t_bgn = t_bgn + t_stp;
%           t_end = t_end + t_stp;
%         end
%         
%         if ~isequal(ref_vec, vec)
%           error('ref_vec and vec not equal');
%         end
%         
        vec = xpnd_mtx.multVec(xpnd_mtx.toFloat(vec));
        
        if last < obj.blk_cnt(3)
          nxt_frm = last+1;
        else
          nxt_frm = [];
        end
      end
      
      function discardFrmsBeforeBlk(obj, blk_indx)
        if isempty(blk_indx)
          % discard all
          removeRawVidBlocker_mex(obj.mex_handle);
          obj.frms_cnt = 0;
          return
        end
        n_dscrd = (blk_indx(3)-1)*(obj.blk_size(1,3)-obj.ovrlp(1,3)) -...
          obj.ovrlp(1,3) - obj.frms_offset;
        removeRawVidBlocker_mex(obj.mex_handle, uint32(blk_indx(1,3)));
        
        obj.frms_offset = obj.frms_offset + n_dscrd;
        obj.frms_cnt = obj.frms_cnt - n_dscrd;
        
        % Code not using MEX
%         if n_dscrd > obj.frms_cnt;
%           n_dscrd = obj.frms_cnt;
%         end
%         if n_dscrd <= 0
%           return;
%         end
%         
%         for k=1:size(obj.blk_size,1)
%           obj.vid_data{k} = obj.vid_data{k}(:,:,n_dscrd+1:end);
%         end

      end
      
    end
    
    methods (Access = private)
      function readBck(obj, n_bck)
        [~, data, err_msg] = read_raw_video(obj.vid_info, n_bck,...
          obj.skip_frms + obj.frms_offset - n_bck + 1);
        if ~isempty(err_msg)
          error('reading input backwards failed (%s)', err_msg);
        end
        
        insertRawVidBlocker_mex(obj.mex_handle, uint32(size(data{1},3)), data, false);
        
        obj.frms_offset = obj.frms_offset - n_bck;
        obj.frms_cnt = obj.frms_cnt + n_bck;

        % Code not using mex
%         for k=1:size(obj.blk_size,1)
%           if isempty(obj.vid_data{k})
%             obj.vid_data{k} = data{k};
%           else
%             obj.vid_data{k} = cat(3, data{k}, obj.vid_data{k});
%           end
%         end
      end
      
      function readFwd(obj, n_fwd, data)
        if nargin < 3
          [~, data, err_msg] = read_raw_video(obj.vid_info, n_fwd,...
            obj.skip_frms + obj.frms_offset + obj.frms_cnt + 1);
          if ~isempty(err_msg)
            error('reading input forwards failed (%s)', err_msg);
          end
        end
        
        insertRawVidBlocker_mex(obj.mex_handle, uint32(size(data{1},3)), data);
        
        obj.frms_cnt = obj.frms_cnt + n_fwd;

        % Code not using mex
%         for k=1:size(obj.blk_size,1)
%           if isempty(obj.vid_data{k})
%             obj.vid_data{k} = data{k};
%           else
%             obj.vid_data{k} = cat(3, obj.vid_data{k}, data{k});
%           end
%         end
      end
    end
end

