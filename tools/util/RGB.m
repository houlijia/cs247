classdef RGB
  % RGB is a class with only static methods, for handling conversion to/from
  % YUV and display of RGB video and images
  %
  % Conversion coefficients from are RGB to YCbCr for HDTV, scaled to the
  % full available range. Based on http://www.equasys.de/colorconversion.html
  
  properties (Constant, Access=private)
    cnvrt_ofst = single([0;128;128]);
    
    to_YUV_mtrx = single(diag(255./[219,224,224]) * [...
      0.183, 0.614, 0.062;...
      -0.101, -0.339, 0.439; ...
      0.439, -0.399, -0.040]);
    
    to_RGB_mtrx = single(inv(RGB.to_YUV_mtrx));
  end
  
  methods (Static)
    function yuv = toYUV(rgb, opts)
      % Convert RGB to YUV
      %   rgb - the RGB input. Can have one of the following forms:
      %         A cell array with 3 cells containing a numerical arrays 
      %           (of same size and class) representing the RGB values.
      %         A numerical vector such that, when reshaped into a 3-column
      %           matrix, each column contains one of the RGB components,
      %           respectively.
      %         A numerical array where one of the dimensions is of size 3 
      %           and the indices 1,2,3 correspond to RGB components.
      %   opts - (optional) a struct with optional fieldd:
      %         dim - in case rgb is a numerical array, the dimension
      %           of the RGB values. non-positive values indicate offset from
      %           the end, eg. 0 indicates the last dimension. Default: 0.
      %         pxmx - Maximal value of a pixel. Default: 255
      %         out_dim - if present, permute the output so that the color
      %             components are along the dim dimension. 0 means
      %             the output should be a 3-cell array, one cell for each
      %             component. default - same as input.
      %         out_sz - output size. If dim==0, size of content of each cell.
      %            Default: same as input
      %         cast - a function handle for casting the output to a
      %            different type than the input.
      % Output:
      %   yuv - the input rgb contverted to yuv, with the same organization
      %         and class as the input.
      
      if nargin < 2
        opts = struct();
      end
      [v, opts] = RGB.normalize(rgb,opts);
      ofst = (opts.scl_fctr*RGB.cnvrt_ofst) * ones(1,size(v,2), 'single');
      u = RGB.to_YUV_mtrx * v + ofst; 
      yuv = RGB.denormalize(u,opts);
    end
    
    function rgb = toRGB(yuv, opts)
      % Convert RGB to YUV
      %   yuv - the YUV input. Can have one of the following forms:
      %         A cell array with 3 cells containing a numerical arrays 
      %           (of same size and class) representing the YUV values.
      %         A numerical vector such that, when reshaped into a 3-column
      %           matrix, each column contains one of the YUV components,
      %           respectively.
      %         A numerical array where one of the dimensions is of size 3 
      %           and the indices 1,2,3 correspond to YUV components.
      %   opts - (optional) a struct with optional fields:
      %         dim - in case rgb is a numerical array, the dimension
      %           of the RGB values. non-positive values indicate offset from
      %           the end, eg. 0 indicates the last dimension. Default: 0.
      %         pxmx - Maximal value of a pixel. Default: 255
      %         out_dim - if present, permute the output so that the color
      %             components are along the dim dimension. 0 means
      %             the output should be a 3-cell array, one cell for each
      %             component. default - same as input.
      %         out_sz - output size. If dim==0, size of content of each cell.
      %            Default: same as input out_sz is ignored if out_dim is
      %            not sepcified.
      %         cast - a function handle for casting the output to a
      %            different type than the input.
      % Output:
      %   rgb - the input yuv contverted to rgb, with the same organization
      %         and class as the input.
      
      if nargin < 2
        opts = struct();
      end
      [v, opts] = RGB.normalize(yuv,opts);
      ofst = (opts.scl_fctr*RGB.cnvrt_ofst) * ones(1,size(v,2), 'single');
      u = RGB.to_RGB_mtrx * (v - ofst); 
      rgb = RGB.denormlize(u,opts);
    end
    
    function [vid,opts] = getVideoArray(vid,opts)
      % Retruns a video array, which is a uint 8 4D-array of size(V,H,3,T)
      % where V,H and the vertical,horizontal number of pixels and T is the
      % number of frames. The thrd dimesnion is the RGB components.
      %   Input:
      %     vid - video specification. Can be one of the following:
      %       A character string, interpreted as the name of the file from
      %         which to read the video.
      %       A struct array with the field 'cdata', which is a 3-D unit8 
      %         array, where the third dimension is RGB color for one
      %         frame.
      %       A vector, which is organized by RGB, 1st third R, 2nd third
      %         G, 3rd third B.
      %       A 4-D matrix for Horizontal, vertical, frame and color
      %         dimensions. By default color is the last dimension, but
      %         this can be overridden.
      %       A cell array or a numerical array containing pixel values
      %     opts - A struct with specification of the input. If fields are
      %       not specified, they are determined from the input if
      %       possible, or get a default value. If the input is a vector, 
      %       opts must be specified and have a sz field.. The fields are:
      %         fps - frames per second. Default: 30;
      %         sz - The dimensions of the video. Needed only when vid is a
      %              vector. Can an row vector of 2 (V,H), 3 (V,H,T) or 4
      %              (V,H,T,C).
      %         dim - in case the input is a numerical array, the dimension
      %           of the color components. non-positive values indicate offset from
      %           the end, eg. 0 indicates the last dimension. Default: 0.
      %         pxmx - Maximal value of a pixel. Default: 255
      %         is_rgb - true if the input is RGB. default: false
      %         n_frames - number of frames to play
      %         skip - number of frames to skip at the beginning
      %   Output:
      %     vid - a 4D uint8 array of RGB values, organzied by (V,H,C,T).
      %     opts - updated opts struct
      
      if nargin < 2
        opts = struct();
      end
      if ~isfield(opts,'skip')
        opts.skip = 0;
      end
      if ~isfield(opts,'n_frames')
        opts.n_frames = inf;
      end
      if ~isfield(opts,'is_rgb')
        opts.is_rgb = false;
      end
      
      if isa(vid,'uint8') &&  opts.is_rgb && ...
        length(size(vid))>2 && opts.dim == 3
        if ~isfield(opts,'sz')
          opts.sz = size(vid);
        end
        if ~isfield(opts,'skip')
          opts.skip = 0;
        end
        if ~isfield(opts,'n_frames')
          opts.n_frames = size(vid,4);
        else
          opts.n_frames = min(opts.n_frames, size(vid,4)-opts.skip);
        end
      
        vid = vid(:,:,:,opts.skip+1:opts.skip+opts.n_frames);
        opts.skip = 0;
        opts.n_frames = size(vid,4);
      elseif ischar(vid)
        [f_info, vid, err_msg] = read_raw_video(vid, opts.n_frames, ...
          opts.skip+1, struct('cast', @(x) single(x)));
        if ~isempty(err_msg)
          error(err_msg);
        end
        opts.n_frames = f_info.seg_n_frames;
        opts.skip = 0;
        if ~isfield(opts,'fps')
          opts.fps = f_info.fps;
        end
        opts.pxmx = f_info.getPixelMax();
        opts.is_rgb = false;
        [vid,opts] = RGB.getVideoArray(vid,opts);
      elseif isstruct(vid) && isfield(vid, 'cdata')
        sz0 = size(vid(1).cdata);
        opts.sz = ones(3,1);
        opts.sz(1:length(sz0)) = sz0;
        opts.sz = [opts.sz numel(cdata)];
        opts.dim = 3;
        v = zeros(opts.sz,'uint8');
        for k=1:size(opts.sz,4)
          v(:,:,:,k) = opts(k).cdata;
        end
        vid = v;
        if ~isfield(opts,'fps')
          opts.fps = 30;
        end
      else
        opts.out_dim = 3;
        opts.cast = @(x) uint8(min(single(255), max(single(0),...
          round(x+single(0.4999)))));
        if ~isfield(opts,'fps')
          opts.fps = 30;
        end

        [v,opts] = RGB.normalize(vid,opts);

        if isfield(opts,'pxmx') && opts.pxmx ~= 255
          % Normalize
          v = v * single(255/opts.pxmx);
        end
        opts.pxmx = 255;
        
        if isfield(opts, 'is_rgb') && ~opts.is_rgb
          % convet to RGB
          ofst = (opts.scl_fctr*RGB.cnvrt_ofst) * ones(1,size(v,2), 'single');
          v = RGB.to_RGB_mtrx * (v - ofst);
        end
        opts.is_rgb = true;

        v = RGB.denormalize(v,opts);
        [vid,opts] = RGB.getVideoArray(v,opts);
      end
    end
    
    function [vid,opts] = showVideo(vid,opts)
      % same as getVideoArray, but also opens a video player window for vid
      if nargin < 2
        opts = struct();
      end
      [vid,opts] = RGB.getVideoArray(vid,opts);
      implay(vid);
    end
    
    function [fg,vid] = playVideo(varargin)
      % Plays video data
      %   Input:
      %     fg - a figure handle - if missing a handle is created.
      %     vid - video specification. Can be one of the following:
      %       A struct array with the fields 
      %         'cdata' a 3-D array, where the third dimension is color for
      %            one frame
      %         'colormap' - can be empty
      %       A vector, which is organized by RGB, 1st third R, 2nd third
      %         G, 3rd third B.
      %       A 4-D matrix for Horizontal, vertical, frame and color
      %         dimensions. By default color is the last dimension, but
      %         this can be overridden.
      %       A cell array or a numerical array containing pixel values
      %       A character string, interpreted as the name of the file from
      %         which to read the video.
      %     rpt - (optional) no. otf times to repeat playing or any array
      %           which is valid input to movie function. Default: 1
      %     opts - A struct with specification of the input. If fields are
      %       not specified, they are determined from the input if
      %       possible, or get a default value. If the input is a vector, 
      %       opts must be specified and have a sz field.. The fields are:
      %         fps - frames per second. Default: 30;
      %         sz - The dimensions of the video. Needed only when vid is a
      %              vector. Can an row vector of 2 (V,H), 3 (V,H,T) or 4
      %              (V,H,T,C).
      %         dim - in case the input is a numerical array, the dimension
      %           of the color components. non-positive values indicate offset from
      %           the end, eg. 0 indicates the last dimension. Default: 0.
      %         pxmx - Maximal value of a pixel. Default: 255
      %         is_rgb - true if the input is RGB. default: false
      %         n_frames - number of frames to play
      %         skip - number of frames to skip at the beginning
      %   Output:
      %     fg - handle to the figure in which the playing is done
      %     vid - A struct array with one entry per played frame, with the fields 
      %         'cdata' a 3-D array, where the third dimension is color for
      %            one frame
      %         'colormap' - can be empty
      
      fg = varargin{1};
      if ~ishandle(fg)
        fig = figure();
        args = [{fig} varargin];
        [fg,vid] = RGB.playVideo(args{:});
        return
      end
      vid = varargin{2};
      if nargin < 3
        rpt = 1;
      else
        rpt = varargin{3};
      end
      if nargin < 4 
        if nargin == 3 && isstruct(varargin{3})
          opts = varargin{3};
          rpt = 1;
        else
          opts = struct();
        end
      else
        opts = varargin{4};
      end
      
      if ~isfield(opts,'skip')
        opts.skip = 0;
      end
      if ~isfield(opts,'n_frames')
        opts.n_frames = inf;
      end
      
      if isstruct(vid)
        if ~isfield(opts,'fps')
          opts.fps = 30;
        end
        vid = vid(opts.skip+1:min(length(vid),opts.skip+opts.n_frames));
        opts.skip = 0;
        opts.n_frames = length(vid);
        
        movie(fg, vid, rpt, opts.fps);
      else
        [v,opts] = RGB.getVideoArray(vid,opts);
        sz = opts.sz(1:3);
        
        vid = struct('cdata', cell(1,size(v,4)), 'colormap', []);
        for k=1:length(vid)
          vid(k).cdata = reshape(v(:,:,:,k),sz);
        end
        
        [fg,vid] = RGB.playVideo(fg,vid,rpt,opts);
      end

    end
  end
  
  methods (Static, Access=private)
    function [v,opts] = normalize(v,opts)
      if ~isfield(opts, 'pxmx')
        opts.pxmx = 255;
      end
      opts.scl_fctr = single(opts.pxmx/255);
      
      if ~isfield(opts, 'cast')
        if iscell(v)
          ref = v{1};
        else
          ref = v;
        end
        if isinteger(ref)
          opts.cast = @(x) cast(round(x+single(0.49999)), class(ref));
        else
          opts.cast = @(x) cast(x, class(ref));
        end
      end
        
      if iscell(v)
        opts.sz = size(v{1});
        opts.dim = 0;  % Indicates that it is a cell array
        w = zeros(3, numel(v{1}), 'single');
        for k=1:3
          w(k,:) = single(v{k}(:));
        end
        v = w;
      elseif isvector(v)
        if isfield(opts, 'sz')
          if length(opts.sz) == 2
            opts.sz = [opts.sz numel(v)/(opts.sz(1)*opts.sz(2)*3)];
          end
          if ~isfield(opts,'dim')
            opts.dim = 4;
          end
          if length(opts.sz) == 3
            opts.sz = [opts.sz(1:opts.dim-1) 3 opts.sz(opts.dim:end)];
          end
          v = reshape(v,opts.sz);
          [v,opts] = RGB.normalize(v,opts);
        else
          opts.sz = [numel(v),1];
          opts.dim = 2;
          v = reshape(single(v), [numel(v)/3, 3]);
          v = v';
        end
      else
        opts.sz = size(v);
        if ~isfield(opts,'dim')
          opts.dim = length(opts.sz);
        elseif opts.dim <= 0
          opts.dim = length(opts.sz) + opts.dim;
        end
        nsz = length(opts.sz);
        d = opts.dim;
        switch d
          case 1
            v = single(reshape(v, [3,numel(v)/3]));
          case nsz
            v = single(reshape(v, [numel(v)/3,3]));
            v = v';
          otherwise
            v = single(permute(double(v), [d, 1:d-1, d+1:nsz]));
            v = reshape(v,[3, numel(v)/3]);
        end
      end
      
      if isfield(opts, 'out_dim') 
        if isfield(opts, 'out_sz')
          opts.sz = opts.out_sz;
        elseif opts.dim ~= opts.out_dim
          if opts.out_dim == 0
            opts.sz = [opts.sz(1:opts.dim-1) opts.sz(opts.dim+1:end)];
          elseif opts.dim == 0
            opts.sz = [opts.sz(1:opts.out_dim-1) 3 opts.sz(opts.out_dim:end)];
          else
            sz = [opts.sz(1:opts.dim-1) opts.sz(opts.dim+1:end)];
            opts.sz = [sz(1:opts.out_dim-1) 3 sz(opts.out_dim:end)];
          end
        end
        opts.dim = opts.out_dim;
      end
    end
    
    function w = denormalize(v,opts)
      if opts.dim == 0; % cell array
        w = cell(1,3);
        for k=1:3
          w{k} = reshape(opts.cast(v(k,:)), opts.sz);
        end
      else
        sz = opts.sz;
        nsz = length(sz);
        d = opts.dim;
        if d == nsz
          v = v';
          w = reshape(opts.cast(v(:)),opts.sz);
        elseif d==1
          w = reshape(opts.cast(v(:)),opts.sz);
        elseif d > 1
          v = reshape(v(:), [sz(d), sz(1:d-1), sz(d+1:nsz)]);
          v = ipermute(double(v), [d, 1:d-1, d+1:nsz]);
          w = opts.cast(v);
        end
        
      end
        
    end
  end
  
end

