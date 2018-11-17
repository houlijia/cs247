classdef VidSparserTV3 < VidSparserTV
  %VidSparserTV3 perform differences in 3 dimensions
  
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
    function obj = VidSparserTV3(args)
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
      obj.ndim = 3;
      obj.set@VidSparserTV(args);
    end
  end
  
end

