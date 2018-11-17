function [ f_info, data, err_msg] = read_raw_video( src_info, n_frames, start_frame )
    %READ_RAW_VIDEO reads raw video (e.g. YUV or BW) from a file.
    %INPUT ARGUMENTS:
    %   src_info: can be either a RawVidInfo object about the data
    %   file, or a string, which is interpreted as the name of file containing
    %   a JSON with the information about the raw video file. In that case the
    %   file will be read and the JSON will be converted to the object.  If the
    %   data file is not open, it is opened.
    %   n_frames: Number of frames requested. If missing read until the end of the file
    %             and close it.
    %   start_frame: Frame number to start at. If missing, start at current
    %   position.
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
    
    if ischar(src_info)
        jstr = fileread(src_info);
        finf = parse_json(jstr);
        f_info = RawVidInfo;
        f_info.calcFrameInfo(finf);
        src_dir = src_info;
    else
        f_info = src_info;
        src_dir = '';
    end
    
    % Open data file if necessary
    if f_info.handle == -1
        err_msg = f_info.openInpFile(src_dir);
        if ~isempty(err_msg)
            return
        end
    end
    
    % Move to start_frame if required
    if nargin < 3
        start_frame = f_info.seg_start_frame + f_info.seg_n_frames+1;
        if nargin == 1
            n_frames = f_info.n_frames - start_frame + 1;
            close_when_done = true;
        end
    end
    if start_frame > f_info.n_frames;
        err_msg='start_frame exceeds end of file';
        return
    elseif start_frame <= 0
        err_msg='start_frame must be positive';
        return
    end
    if nargin == 3
        fseek(f_info.handle, (start_frame-1) * f_info.frame_size, 'bof');
    end
    f_info.seg_start_frame = start_frame-1;
    
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
    for i=1:n_frames
        [Ytrans, cnt] = fread(f_info.handle, size(Ytrans), read_prec);
        if cnt < f_info.Ylen
            err_msg = ferror(f_info.handle);
            break  % Failed to read the whole frame
        end
        if size(data,2) == 3    % Need to read U and V
            [Utrans, cnt] = fread(f_info.handle, size(Utrans), read_prec);
            if cnt < f_info.UVlen
                err_msg = ferror(f_info.handle);
                break  % Failed to read the whole frame
            end
            [Vtrans, cnt] = fread(f_info.handle, size(Vtrans), read_prec);
            if cnt < f_info.UVlen
                err_msg = ferror(f_info.handle);
                break  % Failed to read the whole frame
            end
            
            data{2}(:,:,i)=Utrans';
            data{3}(:,:,i)=Vtrans';
        end
        data{1}(:,:,i)=Ytrans';
        n_frms_read = n_frms_read + 1;
    end
    f_info.n_frames_read = f_info.n_frames_read + n_frms_read;
    f_info.seg_n_frames = n_frms_read;
    
    if close_when_done
        info = f_info;
        [f_info, err_msg] = close_raw_video(info);
        return
    end
    
end

function [ f_info, err_msg ] = close_raw_video( info )
    f_info = info;
    if fclose(f_info.handle) == -1
        emsg = ferror(f_info.handle);
        err_msg = sprintf('Failed closing output file %s (%s)', ...
            f_info.path, emsg);
    else
        err_msg = '';
    end
end



