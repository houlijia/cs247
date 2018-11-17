function [ err_msg, f_handle ] = write_raw_video( dst_info, data, ...
    raw_vid, start_frame)
    %WRITE_RAW_VIDEO write raw video into a file
    %INPUT ARGUMENTS:
    %   dst_info: Specifies the output file.  Can be either a file handle or a
    %             character string specifying the file to write to, in which
    %             case a handle is opened.
    %   data: (optional) The data to be written, as a cell of one (black and white) or three
    %         (YUV) three dimensional arrays. If missing the function simply closes
    %         the file. 
    %   raw_vid: (optional), if present and not empty, it is a RawVidInfo
    %            object which specifies the decimation.
    %   start_frame: (optional), if present, start writing assuming that data
    %                 begins at frame start_frame.
    %OUTPUT ARGUMENTS
    %   err_msg: If not an empty string, an error message
    %   f_handle: (optional) If present returns the file handle. If absent and
    %             the file was specified by a character string, the file is
    %             closed upon exiting.
    
    
    err_msg='';
    if ischar(dst_info)
        dir_dst = fileparts(dst_info);
        if dir_dst
            [success, msg, msgid] = mkdir(dir_dst);
            if ~success
                err_msg = sprintf('Failed creating directory for output file %s (%d: %s)', ...
                    dst_info, msgid, msg);
                return
            end
        end
        [f_handle, emsg] = fopen(dst_info, 'w');
        if f_handle == -1
            err_msg = sprintf('Failed opening output file %s (%s)', ...
                dst_info, emsg);
            return
        end
        close_when_done = (nargout < 2);
    else
        f_handle = dst_info;
    
        % Handle close file case
        if nargin < 2
            if fclose(f_handle) == -1
                emsg = ferror(f_handle);
                err_msg = sprintf('Failed closing output file (%s)',  emsg);
            end
            return
        else
            close_when_done = false;
        end
    
        % Move to start_frame
        if nargin >= 4
            frame_len = 0;
            for k=1:length(data)
                siz = size(data{k});
                frame_len = frame_len + siz(1) * siz(2);
            end
            switch(class(data{1}))
              case {'uint8','int8'}
                frame_size = frame_len;
              case {'uint16','int16'}
                frame_size = frame_len * 2;
              case {'uint32','int32'}
                frame_size = frame_len * 4;
                
              otherwise
                err_msg = sprintf('Unexpected data type: %s', class(data{1}));
                return
            end
            fseek(f_handle, frame_size * (start_frame -1), 'bof');
        end
    end

    % Decimate if necessary
    n_frames = size(data{1},3);
    if ~isempty(raw_vid)
      if isfield(raw_vid.intrplt_mtx,'InvY')
        mtx = SensingMatrixKron.constructKron({...
          SensingMatrixUnit(n_frames), raw_vid.intrplt_mtx.InvY});
        pxls = mtx.multVec(double(data{1}(:)));
        data{1} = reshape(pxls, [raw_vid.height, raw_vid.width, n_frames]);
      end
      
      if isfield(raw_vid.intrplt_mtx,'InvUV') && length(data) > 1
        mtx = SensingMatrixKron.constructKron({...
          SensingMatrixUnit(n_frames), raw_vid.intrplt_mtx.InvUV});
        for k=2:3
          pxls = mtx.multVec(double(data{k}(:)));
          data{k} = reshape(pxls, [raw_vid.UVheight, raw_vid.UVwidth, n_frames]);
        end
      end
    end
    
    for k=1:length(data)
      data{k} = raw_vid.castToPixel(gather(data{k}));
    end
    
    Ylen = size(data{1},1) * size(data{1},2);
    if length(data) == 3
        UVlen = size(data{2},1) * size(data{2},2);
    end
    
    % write out the data
    for i=1:n_frames        
        Ytrans =(data{1}(:,:,i))';
        if fwrite(f_handle, Ytrans) ~= Ylen
            [err_msg, err_num] = ferror(f_handle);
            err_msg = sprintf('Writing failed (%d): %s', err_num, err_msg);
            return
        end
        if length(data) == 3
            Utrans =(data{2}(:,:,i))';
            Vtrans =(data{3}(:,:,i))';
            if fwrite(f_handle, Utrans) ~= UVlen ||...
                    fwrite(f_handle, Vtrans) ~= UVlen
                [err_msg, err_num] = ferror(f_handle);
                err_msg = sprintf('Writing failed (%d): %s', err_num, err_msg);
                return
            end
        end
    end
    
    if close_when_done
        if fclose(f_handle) == -1
            emsg = ferror(f_handle);
            err_msg = sprintf('Failed closing output file (%s)',  emsg);
        end
    end
    
end
