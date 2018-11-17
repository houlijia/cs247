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
        function obj = VidCompare(pxmx, mode, ref, skip, intrplt)
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
          %   intrplt: If true, and the UV components are of lower resolution
          %            than the Y component, the UV components are linearly
          %            interpolated to the resolution of the Y component.
          %            Default = false. This argument is ignored if src_info
          %            is a RawVidInfo object
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
                if nargin >= 4
                  obj.ref_info.setInterpolate(intrplt);
                end
            end
        end
        
        % Add reference frames (in case ref_info is numeric
        function addRef(obj, ref)
           if iscell(ref)
             if isempty(obj.ref_info)
               obj.ref_info = cell(size(ref));
               for k=1:numel(ref);
                 obj.ref_info{k} = double(ref{k});
               end
             else
               for k=1:numel(ref);
                 obj.ref_info{k} = cat(3,obj.ref_info, double(ref{k}));
               end
             end
           else
             if isempty(obj.ref_info)
               obj.ref_info = double(ref);
             else
                obj.ref_info = cat(3,obj.ref_info, double(ref));
             end
           end
        end
        
        % If output is specified returns the PSNR and MSE of the tst_frms
        % only.
        function [psnr, mse] = update(obj, tst_frms, prefix)
            if iscell(tst_frms)
              n_frms = size(tst_frms{1},3);
              Ytst = cell(size(tst_frms));
              for k=1:numel(tst_frms)
                Ytst{k} = double(tst_frms{k});
              end
            else
              n_frms = size(tst_frms,3);
              Ytst = double(tst_frms);
            end
            
            % read reference frames
            if  isnumeric(obj.ref_info)
              Yref = double(obj.ref_info(:,:,1:min(n_frms,size(obj.ref_info,3))));
              obj.ref_info = obj.ref_info(:,:,n_frms+1:end);
            else
              [obj.ref_info, ref_frms, err_msg] = ...
                read_raw_video(obj.ref_info, n_frms);
              if ~isempty(err_msg)
                error(err_msg);
              end
              Yref = cell(size(ref_frms));
              for k=1:numel(Yref)
                Yref{k} = double(ref_frms{k});
              end
            end
            
            % Take care of end of file case, where the number of frames is
            % less than the block number of frames
            if iscell(Yref)
              for k=1:numel(Yref)
                if size(Yref{k},3) < n_frms
                  yr = Yref{k};
                  Yref{k} = zeros(size(Ytst{k}));
                  Yref{k}(:,:,1:size(yr,3)) = yr;
                end
              end
            else
              if size(Yref,3) < n_frms
                yr = Yref;
                Yref = zeros(size(Ytst));
                Yref(:,:,1:size(yr,3)) = yr;
              end
            end
            
            %update
            obj.sqr_err.update(Yref,Ytst);
            
            if nargout > 0 || obj.frm_by_frm
              [psnr,mse] = SqrErr.compPSNR(Yref,Ytst,obj.pxl_max);
              
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
                    imshow(Yref(:,:,i),[]);
                    title(sprintf('%s ref frm %d',prefix, frm_no));
                    subplot(n_frms,2,2*(i-1)+2);
                    imshow(Ytst(:,:,i),[]);
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

