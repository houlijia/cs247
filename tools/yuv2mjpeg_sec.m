function yuv2mjpeg_sec(inp, fname, sec_cnt, sec_ofst, sec_intvl)
%yuv2mjpeg_sec Reads a YUV file and writes out one frame per second as a
%JPEG file.
%  Input:
%    inp - File name of a JSON containing a description of the input video file
%    fname - template name form the output files. Used a format for writing
%            the index number. E.g. If fname is 'out%04d.jpg', the output
%            files will be out0001.jpg, out0002.jpg,...
%    sec_cnt - number of seconds to process (optional, default = inf)
%    sec_ofst - Number of seconds to skip (optional, default = 0).  A value
%               0 means write every frame
%    sec_intvl - Time interval between output frames (optional, default = 1
%                second). A value of 0 means every frame).

    if nargin < 5
        sec_intvl = 1;
        if nargin < 4
            sec_ofst = 0;
            if nargin < 3
                sec_cnt = inf;
            end
        end
    end
    
    f_info = read_raw_video(inp,0);
    frm_cnt = floor(f_info.fps * sec_cnt);
    frm_ofst = ceil(f_info.fps * sec_ofst);
    if sec_intvl
        frm_intvl = f_info.fps * sec_intvl;
    else
        frm_intvl = 1;
    end

    indx = 0;
    while true
        frm_num = round(indx * frm_intvl) + 1;
        if frm_num > frm_cnt
            break;
        end
        frm_num = frm_num + frm_ofst;
        if frm_num > f_info.n_frames
            break;
        end
        
        [f_info, data, err_msg] = read_raw_video(f_info, 1, frm_num);
        if ~isempty(err_msg)
            error('Frame %d indx %d: %s', frm_num, indx, err_msg);
        end
        
        indx = indx+1;
        imwrite(data{1}, sprintf(fname, indx), 'jpeg');
    end
    
end

