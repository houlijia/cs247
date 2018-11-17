classdef VidBlocksOut < VidBlocker
    %VidBlocksOut is a repository for blocks which are generated, combined
    %into video frames and then potentially written out
    
    properties
        % Number of frames that have been written out and no longer
        % available
        n_frms_done = 0; 
        
        % Number of temporal blocks (sets of all blocks corresponding to
        % same time) that have been written and are no longer available
        n_tblks_done = 0;
        
        % Overlap frame from previous writing
        frm_ovrlp = [];
        
        % blocks 4D cell array (C,V,H,T)
        blks = [];
        
        % output file name.  If empty no writing is done
        file_name = [];
        
        % output file ID. If -1 no writing is done
        fid = -1;
        
        % If true, the data is signed and will be centered and shrunk
        is_diff = false;
        
        % processing state of the received blocks (BLK_STT_...)
        blk_stt;
    end
    
    methods
        % Constructor
        %   Input: 
        %     dst: Where output is written. Can be a character string (file
        %          name) or a number (stream number). [] or -1 indicate no
        %          output.
        %     isdff: Value for is_diff
        %     yb_size: Can be the size of the Y component or another
        %          VidBlocker object as an example.  In the latter case the
        %         the last argument (params), if present, is ignroed.
        %     b_stt: The state of the blocks to be written (BLK_STT_...)
        %     params: The params argument in the construction of VidBlocker
        function obj = VidBlocksOut(dst, isdff, yb_size, b_stt, params)
            if isnumeric(yb_size)
                args = { yb_size, params};
            else
                otr = yb_size;
                args = {otr.blk_size(1,:), otr.getParams()};
            end
            
            obj = obj@VidBlocker(args{:});
            
            obj.is_diff = isdff;
            obj.blk_stt = b_stt;
            
            if ischar(dst) && ~isempty(dst);
                obj.file_name = dst;
                obj.fid = fopen(dst,'w');
                if obj.fid == -1
                    error('failed to open file %s', dst);
                end
            elseif isnumeric(dst)
                obj.fid = dst;
            end
            
            % Create a cell array for blocks for one frame
            clr_blk_cnt = obj.calcClrBlkCnt();
            obj.blks = cell(clr_blk_cnt(1:3));
        end
        
        % destructor
        function delete(obj)
          if obj.fid ~= -1
            fclose(obj.fid);
          end
          obj.fid = -1;
        end
        
        % Write out the frames which are ready, i.e. for which all blocks
        % are not empty
        % Input:
        %  obj - this object
        %  rvid - RawVidInfo object specifying the structure of the output
        %        file
        %  max_t_buffer - (optional) maximal number of t_blks to buffer. If
        %                 this number is exceeded empty blocks are written
        %                 as zeros. Default = 2
        % Output:
        %   nfr - If successful, number of frames written out.  otherwise
        %         an error message
        %   vid - returns the output video.  
        function [nfr, vid] = writeReadyFrames(obj, rvid, max_t_buffer)
            if nargin < 3
                max_t_buffer = 2;
            end
            
            n_tblks = 0;
            if ~isempty(obj.blks)
                done = false;
                for t=1:size(obj.blks,4)
                  if size(obj.blks,4) - t >= max_t_buffer
                    n_tblks = n_tblks+1;
                    continue;
                  end
                  for h=1:size(obj.blks,3)
                    for v=1:size(obj.blks,2)
                      if isempty(obj.blks{1,v,h,t})
                        done = true;
                        break
                      end
                    end
                    if done==true
                      break
                    end
                  end
                  if done==true
                    break
                  end
                  
                  n_tblks = n_tblks+1;
                end
            end
            
            if ~n_tblks
                vid = cell(1,size(obj.blk_size,1));
                nfr = 0;
                return;
            end
            
            [nfr, vid, obj.frm_ovrlp] = obj.writeBlocks(obj.fid, ...
                obj.blks(:,:,:,1:n_tblks), rvid, obj.blk_stt,...
                obj.frm_ovrlp, obj.is_diff);
            
             
            if ischar(nfr)
                return;
            end
            
            obj.n_tblks_done = obj.n_tblks_done + n_tblks;
            obj.n_frms_done = obj.n_frms_done +nfr;
            
            obj.blks = obj.blks(:,:,:,n_tblks+1:end);
        end
        
        function insertBlk(obj, blk, blk_indx)
            v=blk_indx(1);
            h=blk_indx(2);
            t=blk_indx(3)-obj.n_tblks_done;
            
            obj.blks(:,v,h,t)=blk;
        end
        
    end
    
end

