classdef VidCompare < handle
    %VidCompare compares a reference video with a test video
    
    properties
        pxl_max;
        frm_by_frm;
        ref_info;
        sqr_err;
        frms_done = 0;
    end
    
    methods
        % Constructor
        %   pxmx - maximum pixel value
        %   mode - select frame by frame comparison:
        %            0: No frame by frame comparison
        %            1: Print frame by frame PSNR
        %            2: Print frame by frame PSNR and display frames next
        %               to each other.
        %   ref - Can be one of the following:
        %         * A RawVidInfo object describing a source file 
        %         * A string, which is interpreted as the name of file
        %           containing JSON string with the information about the
        %           raw video file.
        %         * A cell array, containging frames for later use
        %   skip - (optional) number of frames to skip (default=0).
        %          Relevant only when ref is a string (file name)
        function obj = VidCompare(pxmx, mode, ref, skip)
            if nargin < 4
                skip = 0;
            end
            obj.sqr_err = SqrErr();
            obj.pxl_max = pxmx;
            obj.frm_by_frm = mode;
            
            if isa(ref,'RawVidInfo')
                obj.ref_info = ref;
            elseif iscell(ref)
                obj.ref_info = double(ref{1});
            else
                [f_info,~,err_msg] = read_raw_video(ref, 0, skip+1);
                if ~isempty(err_msg)
                    error('failed to read %s (%s)', ref, err_msg);
                end
                obj.ref_info = f_info;
            end
        end
        
        % Add reference frames (in case ref_info is numeric
        function addRef(obj, ref)
            if isempty(obj.ref_info)
                obj.ref_info = ref{1};
            else
                obj.ref_info = cat(3,obj.ref_info, double(ref{1}));
            end
        end
        
        % If output is specified returns the PSNR and MSE of the tst_frms
        % only.
        function [psnr, mse] = update(obj, tst_frms, prefix)
          if ~iscell(tst_frms)
            tst_frms = {tst_frms};  % make a cell array of 1
          end
          n_frms = size(tst_frms{1},3);
          t_frms = cell(size(tst_frms));
          for k=1:numel(tst_frms)
            t_frms{k} = tst_frms{k}(:);
          end
          tst_vec = double(vertcat(t_frms{:}));
          
          % read reference frames
          if  isnumeric(obj.ref_info)
            ref_frms = {obj.ref_info(:,:,1:n_frms)};
            obj.ref_info = obj.ref_info(:,:,n_frms+1:end);
            ref_vec = ref_frms{1}(:);
          else
            [obj.ref_info, ref_frms, err_msg] = ...
              read_raw_video(obj.ref_info, n_frms);
            if ~isempty(err_msg)
              error(err_msg);
            end
            r_frms = cell(size(ref_frms));
            for k=1:numel(ref_frms)
              r_frms{k} = ref_frms{k}(:);
            end
            ref_vec = double(vertcat(r_frms{:}));
          end
          
          %update
          obj.sqr_err.update(ref_vec,tst_vec);
          
          if nargout > 0 || obj.frm_by_frm
            [psnr,mse] = SqrErr.compPSNR(ref_vec,tst_vec,obj.pxl_max);
            
            if obj.frm_by_frm
              if nargin < 3
                prefix = '';
              end
              fprintf('%s frames %d:%d PSNR=%f\n',...
                prefix, obj.frms_done+1,...
                obj.frms_done+n_frms, psnr);
              
              if obj.frm_by_frm == 2
                figure
                for i=1:n_frms
                  frm_no = obj.frms_done+i;
                  subplot(n_frms,2,2*(i-1)+1);
                  imshow(ref_frms{1}(:,:,i),[]);
                  title(sprintf('%s ref frm %d',prefix, frm_no));
                  subplot(n_frms,2,2*(i-1)+2);
                  imshow(tst_frms{1}(:,:,i),[]);
                  title(sprintf('%s tst frm %d',prefix, frm_no));
                end
              end
            end
          end
          
          obj.frms_done = obj.frms_done + n_frms;
        end
        
        function [psnr,mse] = getPSNR(obj)
          [psnr, mse] = obj.sqr_err.calcPSNR(obj.pxl_max);
        end
        
    end
    
end

