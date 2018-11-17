classdef RawVidInfo < CodeElement
  % RawVidInfo contains information about a video file or a segment of
  % several frames from a video file
  %   Detailed explanation goes here
  
  properties
    % These are the properties which get encoded.
    UVpresent = true;
    prec = 'uint8';
    width;
    height;
    n_frames = 0;  % Number of frames in file. 0 = unknown
    seg_start_frame = 1;  % first frame of this segment of frames
    seg_n_frames = 0;   % number of frames in this segment of frames;
    uv_ratio = [1 1 1];  % Ratio of Y to UV resolution in each dimension
    
    % If not empty, interpolation and uninterpolation matrices for UV.
    % This is a struct with the following potential fields:
    %   UV - interpolation for UV components
    %   Y  - interpolation for Y component
    %   InvUV - inverse interpolation for UV component
    %   InvY - inverse interpolation for Y component
    %   height - height after interpolation
    %   width - width after interpolation
    %   lvl - height and width are divisible by this number
    intrplt_mtx = [];
    
    % These properties are computed from the encoded properties
    path  = '';
    type;
    Ylen;
    UVwidth;
    UVheight;
    UVlen;
    frame_len;         % number of pixels in a frame
    frame_size;        % size of frame in bytes
    
    % These properties are set by the application
    n_frames_read = 0;  % Number of frames read so far
    handle = [];     % for file I/O
    
    % Frames per second
    fps;
    
    % This function handle converts numbers to the pixel format
    castToPixel;
    castToExtPixel
    
  end
  
  properties (Constant)
    precisions = { 'uint8', 'uint16', 'uint32' };
    prec_factor = [1, 2, 4]; % ratio between frame_size and frame_len
    prec_max = [2^8, 2^16, 2^32] - 1;
    prec_convert = {@(x) uint8(x),@(x) uint16(x), @(x) uint32(x)};
  end
  
  methods
    % Constructor.
    % finf is an optional argument. It can be one of the following:
    %   A struct with the fields
    %       fps - frames per second
    %       height - frame height (pixels)
    %       width - frame width (pixels)
    %       path - (optional) YUV file specification. If only the file name
    %              is specified, the YUV file is in the same directory as the
    %              JSON file.
    %       type - (optional) color pixel format and pixel precision. Can
    %              be one of:
    %              'YUV420_8' (default), 'YUV420_16', 'YUV422_8',
    %              'YUV422_16', 'YUV444_8', 'YUV444_16', ...
    %              'BW_8', 'BW_16'
    %     The values in the struct are used to populate the object. If
    %     'path' is present the YUV file is opened.
    %   A string specifying the path to a file.
    %     If the string ends with '.json' the file should contain a JSON
    %       string with the same fields specified above for the case that
    %       finf is a struct. In this case, the JSON file is read into a
    %       struct and the struct is used as explained above. Note that
    %       'path' can contain no directory, in which case the directory
    %       is the same directory as that of the JSON file.
    %     Otherwise, the file is assumed to be a video file of the kind
    %       that Matlab can read. The file is opend with a VideoReader
    %       object and the properties of the VideoReader are used to
    %       calculate the properties of the object.
    function obj = RawVidInfo(finf)
      if nargin < 1
        return
      end
      if isstruct(finf)
        obj.parseFileInfo(finf);
      elseif regexp(finf, '.json$');
        jstr = fileread(finf);
        finfo = parse_json(jstr);
        obj.parseFileInfo(finfo, finf);
      else
        obj.handle = VideoReader(finf);
        vid_inf = obj.handle.get();
        finfo = struct(...
          'fps', vid_inf.FrameRate,...
          'height', vid_inf.Height,...
          'width', vid_inf.Width...
          );
        switch vid_inf.VideoFormat
          case 'RGB24'
            finfo.type = 'YUV444_8';
          case 'RGB48'
            finfo.type = 'YUV444_16';
          case {'Grayscale', 'Mono8'}
            finfo.type = 'BW_8';
          case 'Mono16'
            finfo.type = 'BW_16';
          otherwise
            error('Unknown video format: %s', finfo.videoFormat);
        end
        obj.calcFrameInfo(finfo);
        obj.n_frames = vid_inf.NumberOfFrames;
      end
     end
    
    function delete(obj)
      try
        obj.closeHandle();
      catch clerr;
        rethrow(clerr);
      end
    end
    
%     function eql = isEqual(obj, other)
%       if class(obj) ~= class(other)
%         eql = false;
%         return;
%       end
%       
%       otr = other.copy();
%       otr.handle = obj.handle;
%       otr.path = obj.path;
%       otr.n_frames_read = obj.n_frames_read;
%       otr.seg_start_frame = obj.seg_start_frame;
%       otr.seg_n_frames = obj.seg_n_frames;
%       
%       % Matlab does not report anonymous function handles as equal
%       % unless they are copied.
%       otr.castToPixel = obj.castToPixel;
%       otr.castToExtPixel = obj.castToExtPixel;
%       %         otr.intrplt_mtx = obj.intrplt_mtx;
%       
%       eql = obj.isEqual@CodeElement(otr);
%     end
    
    % Get the maximum of a pixel value
    function pxmax = getPixelMax(obj)
      precision = find(strcmp(obj.prec,obj.precisions),1);
      pxmax = obj.prec_max(precision);
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
      
      if ~isempty(obj.handle)
        obj.closeHandle();
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
    
    function setInterpolate(obj, lvl)
      % If lvl > 0, whenever a frame is read, the UV components are
      % interplolated to the same size as the Y component. Otherwise,
      % they are left at the same size. If lvl>1, all components are
      % interpolated to a size such that in each dimension it is a
      % multiple of lvl.
      
      lvl = uint32(lvl);
      y_frm_sz = [obj.height, obj.width];
      if obj.UVpresent
        uv_frm_sz = [obj.UVheight, obj.UVwidth];
      end
      
      if lvl > 1 && any(mod(uint32(y_frm_sz),lvl))
        obj.intrplt_mtx = struct(...
          'lvl', lvl,...
          'width', lvl * ceil(obj.width/lvl),...
          'height', lvl * ceil(obj.height/lvl));
        y_out_sz = [obj.intrplt_mtx.height, obj.intrplt_mtx.width];
        obj.intrplt_mtx.Y = construct_intrplt_mtx(y_frm_sz, y_out_sz);
        obj.intrplt_mtx.InvY = construct_intrplt_mtx(y_out_sz, y_frm_sz);
        
        if obj.UVpresent
          if ~all(obj.uv_ratio(1:2) == 1)
            obj.intrplt_mtx.UV = construct_intrplt_mtx(uv_frm_sz, y_out_sz);
            obj.intrplt_mtx.InvUV = construct_intrplt_mtx(y_out_sz, uv_frm_sz);
          else
            obj.intrplt_mtx.UV = obj.intrplt_mtx.Y;
            obj.intrplt_mtx.InvUV =  obj.intrplt_mtx.InvY;
          end
        end
      elseif lvl && obj.UVpresent && ~all(obj.uv_ratio(1:2) == 1)
        obj.intrplt_mtx = struct('lvl', lvl);
        obj.intrplt_mtx.UV = construct_intrplt_mtx(uv_frm_sz, y_frm_sz);
        obj.intrplt_mtx.InvUV = construct_intrplt_mtx(y_frm_sz, uv_frm_sz);
      else
        obj.intrplt_mtx = [];
      end
      
      obj.setUseGpu(obj.use_gpu);
      obj.setUseSingle(obj.use_single);
    end
    
    function rt = uvRatio(obj)
      % If UV interpolation is done returns [1 1 1]. Else returns
      % The ratio of resolutions of Y to UV in each dimension
      if ~isempty(obj.intrplt_mtx)
        rt = [1 1 1];
      else
        rt = obj.uv_ratio;
      end
    end
    
    function rt = intrpltRatio(obj)
      % returns a struct with fields Y, UV, giving the combined
      % interpolation ratio for Y and UV respectively.
      rt = struct('Y', 1, 'UV', 1);
      if ~isempty(obj.intrplt_mtx)
        if isfield(obj.intrplt_mtx, 'Y')
          rt.Y = obj.intrplt_mtx.Y.nRows()/obj.intrplt_mtx.Y.nCols();
        end
        if isfield(obj.intrplt_mtx, 'UV')
          rt.UV = obj.intrplt_mtx.UV.nRows()/obj.intrplt_mtx.UV.nCols();
        end
      end
    end
    
    function wd = yWidth(obj)
      if ~isempty(obj.intrplt_mtx) && isfield(obj.intrplt_mtx, 'width')
        wd = obj.intrplt_mtx.width;
      else
        wd = obj.width;
      end
    end
    
    function wd = uvWidth(obj)
      % If UV interpolation is done returns UVwidth. Else returns
      % widty
      if ~isempty(obj.intrplt_mtx)
        wd = obj.yWidth();
      else
        wd = obj.UVwidth;
      end
    end
    
    function ht = yHeight(obj)
      if ~isempty(obj.intrplt_mtx) && isfield(obj.intrplt_mtx, 'height')
        ht = obj.intrplt_mtx.height;
      else
        ht = obj.height;
      end
    end
    
    function ht = uvHeight(obj)
      % If UV interpolation is done returns UVwidth. Else returns
      % widty
      if ~isempty(obj.intrplt_mtx)
        ht = obj.yHeight();
      else
        ht = obj.UVheight;
      end
    end
    
    function fln = frameLen(obj)
      if ~isempty(obj.intrplt_mtx)
        fln = 3*obj.Ylen;
      else
        fln = obj.frame_len;
      end
    end
    
    function len = encode(obj, code_dst, ~)
      if isempty(obj.intrplt_mtx)
        lvl = 0;
      else
        lvl = obj.intrplt_mtx.lvl;
      end
      
      len = code_dst.writeUInt(...
        [obj.n_frames, obj.width, obj.height, ...
        obj.seg_start_frame, obj.seg_n_frames, obj.fps, lvl]);
      if ischar(len); return; end
      
      len1 = code_dst.writeString(obj.type);
      if ischar(len1)
        len = len1;
      else
        len = len + len1;
      end
    end
    
    function len = decode(obj, code_src, ~, cnt)
      if nargin < 4
        cnt = inf;
      end
      
      [vals, len] = code_src.readUInt(cnt, [1,7]);
      if ischar(vals) || (isscalar(vals) && vals == -1)
        len = vals;
        return
      end
      vals = double(vals);
      
      cnt = cnt - len;
      [typ, len1, err_msg] = code_src.readString(cnt);
      if ~ischar(typ)
        if typ == -1;
          len = 'EOD while reading';
        else
          len = err_msg;
        end
        return
      else
        len = len+len1;
      end
      
      obj.parseFileInfo(struct(...
        'fps', vals(6), ...
        'height', vals(3),...
        'width', vals(2),...
        'type', typ));
      obj.n_frames = vals(1);
      obj.seg_start_frame = vals(4);
      obj.seg_n_frames = vals(5);
      obj.setInterpolate(vals(7));  % lvl
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
      
      if obj.use_gpu
        if obj.use_single
         raw_vid{1} = gpuArray.zeros(obj.height, obj.width, n_frms,'single');
          for k=2:n_color
            raw_vid{k} = gpuArray.zeros(obj.uvHeight(), obj.uvWidth(), ...
              n_frms, 'single');
          end
        else
          raw_vid{1} = gpuArray.zeros(obj.height, obj.width, n_frms);
          for k=2:n_color
            raw_vid{k} = gpuArray.zeros(obj.uvHeight(), obj.uvWidth(), ...
              n_frms);
          end
        end
      else
        if obj.use_single
         raw_vid{1} = zeros(obj.height, obj.width, n_frms,'single');
          for k=2:n_color
            raw_vid{k} = zeros(obj.uvHeight(), obj.uvWidth(), n_frms,...
              'single');
          end
        else
          raw_vid{1} = zeros(obj.height, obj.width, n_frms);
          for k=2:n_color
            raw_vid{k} = zeros(obj.uvHeight(), obj.uvWidth(), n_frms);
          end
        end
      end
    end
    
    function closeHandle(~)
      return
      % Disabling closeHandle - gets called unexpectedly in parallel
      % processing.
      %         if isempty(obj.handle)
      %           return
      %         elseif isnumeric(obj.handle)
      %           try
      %             fclose(obj.handle);
      %           catch clerr
      %             if ~strcmp(clerr.identifier,'MATLAB:badfid_mx')
      %               rethrow(clerr);
      %             else
      %               obj.handle = [];
      %             end
      %           end
      %         end
      %         obj.handle = [];
    end
    
  end
  
  methods (Access=protected)
    function otr = copyElement(obj)
      params = struct(...
        'fps', obj.fps, ...
        'height', obj.height,...
        'width', obj.width,...
        'type', obj.type);
      
      otr = RawVidInfo(params);
      otr.n_frames = obj.n_frames;
      otr.path = obj.path;
      if ~isempty(obj.intrplt_mtx)
        otr.setInterpolate(obj.intrplt_mtx.lvl);
      end
      
      % Matlab does not report anonymous function handles as equal
      % unless they are copied.
      otr.castToPixel = obj.castToPixel;
      otr.castToExtPixel = obj.castToExtPixel;
    end
    
    function parseFileInfo(obj, finf, ref_path)
      obj.calcFrameInfo(finf);
      if ~isempty(obj.path)
        if nargin < 3
          err = obj.openInpFile();
        else
          err = obj.openInpFile(ref_path);
        end
        if ~isempty(err)
          error(err);
        end
      end
    end
    
    % Calculate frame size related parameters.  finf is struct
    % having at least the fields 'width', 'height', 'fps' and 'type'
    % and optionally also a 'path' field.
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
          obj.UVpresent = false;
        otherwise
          err = sprintf('Unknown type: %s',obj.type);
          return
      end
      switch(obj.type)
        case {'YUV420_8', 'YUV422_8', 'YUV444_8', 'BW_8'}
          obj.frame_size = obj.frame_len;
          obj.prec = 'uint8';  % Reading precision
          obj.castToPixel = @(x) uint8(x);
          obj.castToExtPixel = @(x) int32(x);
        otherwise
          obj.frame_size = obj.frame_len *2;
          obj.prec = 'uint16';  % Reading precision
          obj.castToPixel = @(x) uint16(x);
          obj.castToExtPixel = @(x) int32(x);
      end
    end
    
    function setUseGpu(obj,val)
      if isempty(obj.intrplt_mtx)
        return
      end
      nms = {'Y', 'invY', 'UV', 'invUV'};
      for k=1:length(nms)
        nm = nms{k};
        if isfield(obj.intrplt_mtx, nm)
          obj.intrplt_mtx.(nm).use_gpu = val;
        end
      end
    end
    
    function setUseSingle(obj,val)
      if isempty(obj.intrplt_mtx)
        return
      end
      nms = {'Y', 'invY', 'UV', 'invUV'};
      for k=1:length(nms)
        nm = nms{k};
        if isfield(obj.intrplt_mtx, nm)
          obj.intrplt_mtx.(nm).use_single = val;
        end
      end
    end
    
  end
  
  methods (Static, Access=protected)
    function ign = ignoreInEqual()
      ign = [CodeElement.ignoreInEqual() ...
        {'handle', 'path', 'n_frames_read', 'seg_start_frame', 'seg_n_frames'...'
        'castToPixel', 'castToExtPixel'}];
    end
  end
end

