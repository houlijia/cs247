classdef VidSparserD3 < BaseSparser
  %SparserD3 computes for each point, except along the edges 3
  %consecutive differences along vertical, horizontal and temporal
  %dimensions.
  
  properties
    % pixelization parameters for vertical and horizontal differences
    diff_pxl_prms;
    
    % way of computing differences with VidRegion.compDiff
    %            0 - inside differnces - the output dimension is
    %                smaller by one from the input dimentsion.
    %            1 - circular differences - the output dimension
    %                is the same as the input dimesnion.
    %            2 - extended - zero is added before the entries,
    %                so the first entry is preserved.
    mode=0;
  end
  
  methods
    % Constructor
    %     args - a struct with fields necessary to create the sparsifier.
    %            at minimum this should be:
    %              vdrg - The VidRegion object
    %              expand_level - expansion level of the input to the
    %                             sparsifier
    %              expander - the expander used to multiply the input
    %                         vector before sparsifying
    %              unexpander - (optional) inverse of the expander. If
    %                           present, normalization is done.
    %              mode - Way of computing differences with VidRegion.compDiff
    %               0 - inside differnces - the output dimension is
    %                   smaller by one from the input dimentsion.
    %               1 - circular differences - the output dimension
    %                   is the same as the input dimesnion.
    %               2 - extended - zero is added before the entries,
    %                   so the first entry is preserved.
    function obj = VidSparserD3(args)
      if nargin > 0
        obj.set(args);
      end
    end
    
    % Perform the actual operations of the constructor. Has the same
    % arguments.
    %     args - a struct with fields necessary to create the sparsifier.
    %            at minimum this should be:
    %              vdrg - The VidRegion object
    %              expand_level - expansion level of the input to the
    %                             sparsifier
    %              expander - the expander used to multiply the input
    %                         vector before sparsifying
    %              unexpander - (optional) inverse of the expander. If
    %                           present, normalization is done.
    %              mode - Way of computing differences with VidRegion.compDiff
    %               0 - inside differnces - the output dimension is
    %                   smaller by one from the input dimentsion.
    %               1 - circular differences - the output dimension
    %                   is the same as the input dimesnion.
    %               2 - extended - zero is added before the entries,
    %                   so the first entry is preserved.
    function set(obj, args)
      if isfield(args,'mode')
        obj.mode = args.mode;
      end
      obj.set@BaseSparser(args);
    end
    
    function args = getArgs(obj) 
      args = obj.getArgs@BaseSparser();
      args.mode = obj.mode;
    end
    
    function setWgts(obj, wts)
      wts = obj.vid_region.pixelize(wts, obj.blk_stt);
      for dim = 1:3
        wts = VidRegion.getDiffPxls(wts);
      end
      obj.wgts =  VidRegion.vectorize(wtsd);
    end
    
    
  end
  
  methods (Access=protected)
    function setBaseMtx(obj, ~)
      prms = obj.vid_region.getParams_vecToBlks(obj.blk_stt);
      obj.base_mtx = VidRegion.getDiff3Mtrx(prms, obj.mode);
    end
  end
end

