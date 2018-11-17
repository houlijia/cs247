function [ f_info, data, err_msg] = read_raw_video( src_info, ...
    n_frames, start_frame, params)
    %READ_RAW_VIDEO reads raw video (e.g. YUV or BW) from a file.
    %INPUT ARGUMENTS:
    %   src_info: can be either a RawVidInfo object about the data
    %   file, or a string. If the string ends with '.json' it is interpreted
    %   as the name of file containing a JSON string with information about
    %   the YUV video file to be read. The expected fields in the JSON file
    %   are:
    %     fps - frames per second
    %     height - frame height (pixels)
    %     width - frame width (pixels)
    %     path - (optional) file specification. If only the file name is
    %            specified, the YUV file is in the same directory as the
    %            JSON file.
    %     type - color pixel format and pixel precision. Can be one of:
    %            'YUV420_8' (default), 'YUV420_16', 'YUV422_8', 'YUV422_16',
    %            YUV444_8', 'YUV444_16', 'BW_8', 'BW_16',
    %   In this case, the file will be read and the JSON string is
    %   converted to a RawVidInfo object.
    %
    %   If the string does not end with a different extension, it is
    %   assumed to be a file name in a format that Matlab understands (e.g.
    %   AVI). In this case the file is opened and RawVidInfo is returned.
    %   file will be read and the JSON will be converted to the object.  
    %
    %   If src_info is a RawVidInfo object and the data file is not open, 
    %   it is opened.
    %
    %   n_frames: Number of frames requested. If missing read until the end
    %             of the file and close it.
    %
    %   start_frame: Frame number to start at. If missing, start at current
    %   position.
    %
    %   params: An optional struct argument which specifies processing
    %         options. Can have the following fields:
    %       cast: an optional argument which specifies a cast function
    %         (   function handle) to which the output is cast. Empty means
    %         no cast (default)
    %       intrplt: If true, whenever a frame is read, the UV components 
    %                (if present) are interplolated to the same size as the
    %                Y component. Otherwise, they are left at the same size.
    %                If params.intrplt>1, all components are interpolated 
    %                to a size such that in each dimension it is a
    %                multiple of params.intrplt. Default: 0.
    
    
    %OUTPUT ARGUMENTS
    %   f_info: updated RawVidInfo object containing information about the input
    %   file (includingile handle and size).
    %   data:  The data read as a cell of one (black and white) or three
    %   (YUV) three dimensional arrays.  The elements of the arrays are integers of the
    %   smallest size necessary for keeping the data.
    %   err_msg: If not an empty string, an error message
    
    err_msg = '';
    data = [];
    close_when_done = false;
    
    % Move to start_frame if required
    if nargin < 4
      params = struct();
    end
    
    if ~isfield(params, 'cast')
      params.cast = [];
    end
    if ~isfield(params, 'intrplt')
      params.intrplt = 0;
    end
    
    if ~isa(src_info, 'RawVidInfo')
      f_info = RawVidInfo(src_info);
      f_info.setInterpolate(params.intrplt);
    else
      f_info = src_info;
    end
    
    if nargin < 3
      start_frame = f_info.seg_start_frame + f_info.seg_n_frames+1;
      if nargin == 1
        n_frames = f_info.n_frames - start_frame + 1;
        if ~isa(src_info, 'RawVidInfo')
          close_when_done = true;
        end
      end
    end
    
    if start_frame > f_info.n_frames;
        err_msg='start_frame exceeds end of file';
        return
    elseif start_frame <= 0
        err_msg='start_frame must be positive';
        return
    end
    
    % Adjust number of frames to what is available
    if start_frame -1 + n_frames > f_info.n_frames
        n_frames = f_info.n_frames - start_frame + 1;
    end
    % prepare to read. Data in stored in YUV files row by row (as raster
    % scanning is).  When Matlab reads data it is stored in a matrix column by
    % column, hence we need to transpose each frame.
    n_frms_read = 0;
    read_prec = ['*',f_info.prec];
    if f_info.UVpresent
        data = cell(1,3);
    else
        data = cell(1,1);
    end
    data{1}=zeros([f_info.height, f_info.width, n_frames], f_info.prec);
    Ytrans = zeros([f_info.width, f_info.height], f_info.prec);
    if size(data,2) == 3
        Utrans = zeros([f_info.UVwidth, f_info.UVheight], f_info.prec);
        Vtrans = zeros([f_info.UVwidth, f_info.UVheight], f_info.prec);
    end
    
    % Do the actual reading
    if ~n_frames
      n_frms_read = 0;
    elseif isnumeric(f_info.handle)
      old_ofst = f_info.seg_start_frame + f_info.seg_n_frames;
      new_ofst = start_frame-1;
      if  old_ofst ~= new_ofst
        err = fseek(f_info.handle, ...
          (new_ofst-old_ofst)*f_info.frame_size, 'cof');
        if err
          err_msg = ferror(f_info.handle);
          return
        end
      end
          
      for i=1:n_frames
        [Ytrans, cnt] = fread(f_info.handle, size(Ytrans), read_prec);
        if cnt < f_info.Ylen
          err_msg = ferror(f_info.handle);
          return  % Failed to read the whole frame
        end
        if size(data,2) == 3    % Need to read U and V
          [Utrans, cnt] = fread(f_info.handle, size(Utrans), read_prec);
          if cnt < f_info.UVlen
            err_msg = ferror(f_info.handle);
            return  % Failed to read the whole frame
          end
          [Vtrans, cnt] = fread(f_info.handle, size(Vtrans), read_prec);
          if cnt < f_info.UVlen
            err_msg = ferror(f_info.handle);
            return  % Failed to read the whole frame
          end
          
          data{2}(:,:,i)=Utrans';
          data{3}(:,:,i)=Vtrans';
        end
        data{1}(:,:,i)=Ytrans';
        n_frms_read = n_frms_read + 1;
      end
      
      if ~isempty(f_info.intrplt_mtx)
        if isfield(f_info.intrplt_mtx, 'Y')
          mtx = SensingMatrixKron.constructKron({...
            SensingMatrixUnit(n_frames), f_info.intrplt_mtx.Y});
          pxls = reshape(mtx.multVec(mtx.toFloat(data{1}(:))), ...
            [f_info.intrplt_mtx.height, f_info.intrplt_mtx.width, n_frames]);
          if isempty(params.cast)
            data{1} = f_info.castToPixel(round(pxls));
          else
            data{1} = pxls;
          end
        elseif f_info.use_gpu
          data{1} = gpuArray(data{1});
        end
        if isfield(f_info.intrplt_mtx, 'UV')
          mtx = SensingMatrixKron.constructKron({...
            SensingMatrixUnit(n_frames), f_info.intrplt_mtx.UV});
          for k=2:3
            pxls = reshape(mtx.multVec(mtx.toFloat(data{k}(:))), size(data{1}));
            if isempty(params.cast)
              data{k} = f_info.castToPixel(round(pxls));
            else
              data{k} = pxls;
            end
          end
        elseif f_info.use_gpu
          for k=1:3
            data{k} = gpuArray(data{k});
          end
        end
      elseif f_info.use_gpu
        for k=1:3
          data{k} = gpuArray(data{k});
        end
      end
      
      if ~isempty(params.cast)
        for k=1:numel(data)
          data{k} = params.cast(data{k});
        end
      end
    else
      vals = read(f_info.handle, [start_frame, start_frame+n_frames-1]);
%       szv1 = size(vals);
%       szv = ones(1,4);
%       szv(1:length(szv1)) = szv1;
%       szo = szv([1,2,4]);
          
      n_frms_read = size(vals,4);

      if size(data,2) == 3
        opts = struct('dim',3,'pxmx', f_info.getPixelMax(), 'out_dim', 0);
        if ~isempty(params.cast)
          opts.cast = params.cast;
        end
        data = RGB.toYUV(vals, opts);
        
%         pmt = [1,2,4,3];
%         pxmx = f_info.getPixelMax();
%         rgb = single(permute(vals,pmt));
%         yuv = vecRGB2YUV(rgb(:), pxmx);
%         yuv = max(single(0.49999),min(single(pxmx+.49999),yuv));
%         yuv = f_info.castToPixel(round(yuv));
        
%         scl_fctr = single(256/(f_info.getPixelMax()+1));
%         % Conversion coefficients from RGB to YCbCr for HDTV, scaled to the
%         % full available range. Based on http://www.equasys.de/colorconversion.html
%         cnvrt_mtrx = single(diag(255./[219,224,224]) * [...
%           0.183, 0.614, 0.062;...
%           -0.101, -0.339, 0.439; ...
%           0.439, -0.399, -0.040]);
%         cnvrt_ofst = single([0;128;128]) * ones(1,numel(vals)/3,'single');
%         rgb = permute(vals,[3,1,2,4]);
%         rgb_col = single(reshape(rgb,[3, numel(rgb)/3]))*scl_fctr;
%         yuv = max(single(0.5),min(single(255.49999),...
%           (cnvrt_mtrx * rgb_col + cnvrt_ofst)));
%         yuv = f_info.castToPixel(round(yuv/scl_fctr));

%         for k=1:3
%           data{k} = reshape(yuv(:,k),szo);
%         end 
      else % BW
        data{1} = reshape(vals, szv([1,2,4]));
      end
    end
    f_info.n_frames_read = f_info.n_frames_read + n_frms_read;
    f_info.seg_start_frame = start_frame-1;
    f_info.seg_n_frames = n_frms_read;
    
    if close_when_done
      f_info.closeHandle();
    end
    
end


