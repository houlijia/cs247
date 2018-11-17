classdef VidSparserTV_DCT < VidSparserTV
  % VidSparserTV_DCT Same sparser as VidSparserTV, except that the sparsing
  % operation includes a temporal DCT before the difference computation.
  %
  
  properties
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
    function obj = VidSparserTV_DCT(args)
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
      obj.t_trnsfrm = @VidSparserTV_DCT.doDCT;
      obj.inv_t_trnsfrm = @VidSparserTV_DCT.doIDCT;
      obj.set@VidSparserTV(args);
    end
    
    function setWgts(obj, wts)
      wts = obj.vid_region.pixelize(wts, obj.blk_stt);
      for k=1:numel(wts)
        blk = wts{k};
        sz = size(blk);
        blkmn = mean(blk, 3);
        blk = blkmn(:) * ones(1, sz(3));
        wts{k} = reshape(blk, sz);
      end
      wts = obj.vid_region.vectorize(wts);
      obj.setWgts@VidSparserTV(wts);
    end
    
  end
  
  methods (Static)
    function x = doDCT(x)
      if size(x,1) > 1
        y = dct(x);
        x(:) = y(:); % Make sure the output is the same type as the input
      end
    end
    function x = doIDCT(x)
      if size(x,1) > 1
        y = idct(x);
        x(:) = y(:); % Make sure the output is the same type as the input
      end
    end
  end
  
end

