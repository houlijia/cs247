classdef RawVidInfo < CodeElement
    % RawVidInfo contains information about a video file or a segment of
    % several frames from a video file
    %   Detailed explanation goes here
    
    properties
        % These are the properties which get encoded.
        UVpresent = true;
        prec = 'uint8'
        width = 352;    % CIF is the default
        height = 288;
        n_frames = 0;  % Number of frames in file. 0 = unknown
        seg_start_frame = 1;  % first frame of this segment of frames
        seg_n_frames = 0;   % number of frames in this segment of frames;
        uv_ratio = [2 2 1];  % Ratio of Y to UV resolution in each dimension
        
        % These properties are computed from the encoded properties
        path  = '';
        type = 'YUV420_8';
        Ylen = 352 * 288;
        UVwidth = 352/2;
        UVheight = 288/2;
        UVlen = (352/2)*(288/2);
        frame_len;         % number of pixels in a frame
        frame_size;        % size of frame in bytes
        
        % These properties are set by the application
        n_frames_read = 0;  % Number of frames read so far
        handle = -1;     % for file I/O
        
        % Frames per second
        fps;
    end
    
    properties (Constant)
        precisions = { 'uint8', 'uint16', 'uint32' };
        prec_factor = [1, 2, 4]; % ratio between frame_size and frame_len
        prec_max = [2^8, 2^16, 2^32] - 1;
        prec_convert = {@(x) uint8(x),@(x) uint16(x), @(x) uint32(x)}; 
    end
        
    methods
       function eql = isEqual(obj, other)
            if class(obj) ~= class(other)
                eql = false;
                return;
            end
            
            otr = other.copy();
            otr.code = obj.code;
            otr.handle = obj.handle;
            otr.path = obj.path;
            otr.n_frames_read = obj.n_frames_read;
            
            eql = obj.isEqual@CodeElement(otr);
       end
       
       % Calculate frame size related parameters.  finf is struct
        % having at least the fields 'width' and 'height', fps,
        % and optionally also a 'path' and/or 'type' fields
        % err returns an empty string for success or an error code
        function err = calcFrameInfo(obj, finf)
            err = '';
            obj.width = finf.width;
            obj.height = finf.height;
            obj.fps = finf.fps;
            if isfield(finf, 'path')
                obj.path = finf.path;
            end
            if isfield(finf, 'type');
               obj.type = finf.type;
            end
            obj.Ylen = obj.width * obj.height;
            switch(obj.type)
                case {'YUV420_8', 'YUV420_16', 'YUV422_8', 'YUV422_16', ...
                        'YUV444_8', 'YUV444_16'}
                    switch(obj.type)
                        case {'YUV420_8', 'YUV420_16'}
                            obj.uv_ratio = [2 2 1];
                        case {'YUV422_8', 'YUV422_16'}
                            obj.uv_ratio = [1 2 1];
                        otherwise
                            obj.uv_ratio = [1 1 1];
                    end
                    obj.UVheight = obj.height/obj.uv_ratio(1);
                    obj.UVwidth = obj.width/obj.uv_ratio(2);
                    obj.UVlen = obj.UVwidth * obj.UVheight;
                    obj.frame_len = obj.Ylen + 2*obj.UVlen;
                case {'BW_8', 'BW_16'}
                    obj.frame_len = obj.Ylen;
                otherwise
                    err = sprintf('Unknown type: %s',obj.type);
                    return
            end
            switch(obj.type)
                case {'YUV420_8', 'YUV422_8', 'YUV444_8', 'BW_8'}
                    obj.frame_size = obj.frame_len;
                    obj.prec = 'uint8';  % Reading precision
                otherwise
                    obj.frame_size = obj.frame_len *2;
                    obj.prec = 'uint16';  % Reading precision
            end
        end
        
        % Get a struct with the fields width, height, path, type.
        function finf = getFields(obj)
            finf = struct('type', obj.type, 'path', obj.type,...
                'height', obj.height, 'width', obj.width);
        end
        
        function bsize = getBlkSize(obj)
            if obj.UVpresent
                bsize = [obj.height, obj.width, obj.seg_n_frames;...
                    obj.UVheight, obj.UVwidth, obj.seg_n_frames;...
                    obj.UVheight, obj.UVwidth, obj.seg_n_frames];
            else
                bsize = [obj.height, obj.width, obj.seg_n_frames];
            end
        end
        
        % Get the maximum of a pixel value
        function pxmax = getPixelMax(obj)
            precision = find(strcmp(obj.prec,obj.precisions),1);
            pxmax = obj.prec_max(precision);
        end
        
        % Convert valuse to the write precision
        %   Input:
        %     obj - this object
        %     vals - an array of object of a numerical type
        %   Output
        %     pxls - vals converted to the class of the pixels
        function pxls = convertValsToPxls(obj, vals)
            precision = find(strcmp(obj.prec,obj.precisions),1);
            pxls = obj.prec_convert{precision}(vals);
        end
            
        % Open and input data file and set the handle.  
        % The file name is obj.path and the handle is set to obj.handle.
        % ref_path is an optional file path.  If present and obj.path has only a
        % file name without a directory, the director from ref_path is prepended of
        % obj.path
        % err is '' if successful or an error code
        function err = openInpFile(obj, ref_path)
            err = '';
            fname = obj.path;
            if nargin > 1
                src_dir = fileparts(ref_path);
                fdir = fileparts(fname);
                if strcmp(fdir,'')
                    fname = fullfile(src_dir, fname);
                end
            end
            
            if obj.handle ~= -1
                if fclose(obj.handle) == -1
                    err = sprintf('Failed closing input handle(%s)',...
                        ferror(obj.handle));
                    return;
                else
                    obj.handle = -1;
                end
            end
            [obj.handle, emsg] = fopen(fname, 'r');
            if obj.handle == -1
                err = ['Failed opening ' fname ' - ' emsg];
                return
            end
            fseek(obj.handle, 0, 'eof');
            obj.n_frames = floor(ftell(obj.handle)/obj.frame_size);
            fseek(obj.handle, 0, 'bof');
            obj.n_frames_read = 0;
        end
        
        function len = encode(obj, code_dst, ~)
            precision = find(strcmp(obj.prec,obj.precisions),1);
            if isempty(precision)
                len = 'precision is undefined';
                return
            end
            
            len = code_dst.writeUInt(...
                [obj.UVpresent, precision, obj.n_frames...
                obj.width, obj.height...
                obj.seg_start_frame, obj.seg_n_frames, obj.fps]);
            if ischar(len); return; end
            
            if obj.UVpresent
                len1 = code_dst.writeUInt(obj.uv_ratio);
                if ischar(len1);len = len1; return; end
                len = len + len1;
            end
        end
        
        function len = decode(obj, code_src, ~, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            [vals, len] = code_src.readUInt(cnt, [1,8]);
            if ischar(vals) || (isscalar(vals) && vals == -1)
                len = vals;
                return
            end
            vals = double(vals);
            
            obj.UVpresent = vals(1);
            precision = vals(2);
            obj.prec = obj.precisions{precision};
            obj.n_frames = vals(3);
            obj.width = vals(4);
            obj.height = vals(5);
            obj.seg_start_frame = vals(6);
            obj.seg_n_frames = vals(7);
            obj.fps = vals(8);

            obj.Ylen = obj.width * obj.height;

            if obj.UVpresent
                cnt = cnt - len;
                [vals, len1] = code_src.readUInt(cnt, [1,3]);
                if ischar(vals)
                    len = vals;
                    return
                elseif isscalar(vals) && vals == -1
                    len = 'EOD encountered while reading';
                    return
                else
                    vals = double(vals);
                    len = len + len1;
                end
                obj.uv_ratio = vals;
                obj.UVheight = obj.height/vals(1);
                obj.UVwidth = obj.width/vals(2);
                if vals(2) == 1
                  typ = 'YUV444_';
                elseif vals(1) == 1
                  typ = 'YUV422_';
                else
                  typ = 'YUV420_';
                end
                
                obj.UVlen = obj.UVwidth * obj.UVheight;
                obj.frame_len = obj.Ylen + 2*obj.UVlen;
            else
                typ = 'BW_';
                obj.frame_len = obj.Ylen;
            end
            
            obj.frame_size = obj.frame_len * obj.prec_factor(precision);
            obj.type = [typ obj.prec(5:end)];

        end
        
        function raw_vid  = createEmptyVideo(obj, n_frms)
            if nargin < 2
                n_frms = obj.seg_n_frames;
            end
            
            if obj.UVpresent
                n_color = 3;
            else
                n_color = 1;
            end
            
            raw_vid = cell(1,n_color);
            raw_vid{1} = zeros(obj.height, obj.width, n_frms);
            for k=2:n_color
                raw_vid{k} = zeros(obj.UVheight, obj.UVwidth, n_frms);
            end
        end
    end
end

