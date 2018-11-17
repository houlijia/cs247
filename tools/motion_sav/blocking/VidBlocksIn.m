classdef VidBlocksIn < VidBlocker
    %VidBlocksIn Enables reading blocks from a file, while keeping
    %   a buffer of only necessary frames
    
    properties
        % frames available in buffer
        frms_cnt=0;
        frms_offset=0;
        
        % VidRawInfo object for the input video file
        vid_info = [];
        
        % Video data (cell array) of frms_cnt frames, beginning at frame
        % skip_frms+frms_offset+1
        vid_data;
    end
    
    properties (Access = private)
        % Allowed range in file
        skip_frms=0;

        close_on_exit = false;
    end
    
    methods
        % Constructor
        %   yb_size - Size of Y block
        %   params - A params struct suitalbe for the constructor
        %            of a VidBlocker, or a VidBlocker object as
        %            an example.  The field vid_size will be supplied by
        %            this function and the field monochrom will be set if
        %            the file is such.
        %   src - A path to a file containing a JSON descriptor of the
        %         input or a RawVidInfo object
        %   nfrm - (optional) maximal number of frames to read. Default,or
        %          if nfrm==-1: no. of frames in the file.
        %   skip - (optional) Number of frames to skip from the
        %           beginning of the file. Default=0.
        function obj = VidBlocksIn(yb_size, params, src, nfrm, skip)
            if nargin < 5
                skip =0;
                if nargin < 4 || nfrm == -1
                    nfrm = inf;
                end
            elseif nfrm == -1
                nfrm = inf;
            end
            
            if ~isa(src, 'RawVidInfo')
                [f_info, ~, err_msg] = read_raw_video(src, 0, skip+1);
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
            elseif src.n_frames < nfrm+skip
                nfrm = src.n_frames - skip;
            end
            
            if isa(params, 'VidBlocker')
                params = params.getParams();
            end
            if ~src.UVpresent || params.monochrom
                params.monochrom = true;
            	params.vid_size = [src.height, src.width, nfrm];
            else
            	params.vid_size = [src.height, src.width, nfrm;...
                    src.UVheight, src.UVwidth, nfrm;...
                    src.UVheight, src.UVwidth, nfrm];
            end
            
            params.fps = src.fps;
            
            obj = obj@VidBlocker(yb_size, params);
            
            if ~all(obj.blk_size(:))
                error('zero entries in blk_size');
            end
            obj.vid_info = src;
            obj.vid_data = cell(size(obj.blk_size,1),1);
            obj.skip_frms = skip;
            obj.close_on_exit = clex;
        end
        
        function delete(obj)
            if obj.close_on_exit && obj.vid_info.handle ~= -1
                fclose(obj.vid_info.handle);
            end
        end
        
        function blk = getSingleBlk(obj, ~, blk_indx)
            [orig, blk_end, ~, ~, ~, blk_ofst] = obj.blkPosition(blk_indx);
            
            if obj.frms_cnt > 0
                n_bck = obj.frms_offset - orig(1,3);
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
            
            orig(:,3) = orig(:,3) - obj.frms_offset;
            blk_end(:,3) = blk_end(:,3) - obj.frms_offset;
            
            blk = obj.do_getSingleBlk(obj.vid_data, blk_indx, ...
                orig, blk_end, blk_ofst);
        end
        
        function [blk, nxt_blk_indx] = getBlks(obj, blk_indx)
            if nargin < 2
                blk_indx = [];
            end
            
            [blk, nxt_blk_indx] = obj.getBlk([], blk_indx);
        end
        
        function discardFrmsBeforeBlk(obj, blk_indx)
            n_dscrd = (blk_indx(3)-1)*(obj.blk_size(1,3)-obj.ovrlp(1,3)) -...
                obj.ovrlp(1,3) - obj.frms_offset;
            
            if n_dscrd > obj.frms_cnt;
                n_dscrd = obj.frms_cnt;
            end
            if n_dscrd <= 0
                return;
            end
            
            for k=1:size(obj.blk_size,1)
                obj.vid_data{k} = obj.vid_data{k}(:,:,n_dscrd+1:end);
            end
            obj.frms_offset = obj.frms_offset + n_dscrd;
            obj.frms_cnt = obj.frms_cnt - n_dscrd;
        end
        
    end
    
    methods (Access = private)
        function readBck(obj, n_bck)
            [~, data, err_msg] = read_raw_video(obj.vid_info, n_bck,...
                obj.skip_frms + obj.frms_offset - n_bck + 1);
            if ~isempty(err_msg)
                error('reading input backwards failed (%s)', err_msg);
            end
            
            for k=1:size(obj.blk_size,1)
                if isempty(obj.vid_data{k})
                    obj.vid_data{k} = data{k};
                else
                    obj.vid_data{k} = cat(3, data{k}, obj.vid_data{k});
                end
            end
            obj.frms_offset = obj.frms_offset - n_bck;
            obj.frms_cnt = obj.frms_cnt + n_bck;
        end
        
        function readFwd(obj, n_fwd)
            [~, data, err_msg] = read_raw_video(obj.vid_info, n_fwd,...
                obj.skip_frms + obj.frms_offset + obj.frms_cnt + 1);
            if ~isempty(err_msg)
                error('reading input forwards failed (%s)', err_msg);
            end
            
            for k=1:size(obj.blk_size,1)
                if isempty(obj.vid_data{k})
                    obj.vid_data{k} = data{k};
                else
                    obj.vid_data{k} = cat(3, obj.vid_data{k}, data{k});
                end
            end
            obj.frms_cnt = obj.frms_cnt + n_fwd;
        end
    end
    
end

