classdef VidSparserTV_DCT < VidSparserTV
    % VidSparserTV_DCT Same sparser as VidSparserTV, except that the sparsing
    % operation includes a temporal DCT before the difference computation.
    % 
    
    properties
    end
    
    methods
        % Constructor
        %   Input:
        %      vdrg - a (handle of a) VidRegion object
        %      mode - way of computing differences with VidRegion.compDiff
        %            0 - inside differnces - the output dimension is
        %                smaller by one from the input dimentsion.
        %            1 - circular differences - the output dimension
        %                is the same as the input dimesnion.
        %            2 - extended - zero is added before the entries,
        %                so the first entry is preserved.
        %      expander - if present, expander matrix
        function obj = VidSparserTV_DCT(varargin)
          obj.set(varargin{:});
        end
    
        % Apply the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sgnl      - The input signal (of dimension n_sigvec)
        %  Output
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        function sprs_vec = do_compSprsVec(obj, sgnl)
            u = obj.vid_region.comp1dTrnsfrm(sgnl, 3, @(X) mtrx_dct(X));
            sprs_vec = obj.do_compSprsVec@VidSparserTV(u);
        end

        % Apply the transpose of the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        %  Output
        %     sgnl      - The input signal (of dimension n_sigvec)
        function sgnl = do_compSprsVecTrnsp(obj, sprs_vec)
            sgnl = obj.do_compSprsVecTrnsp@VidSparserTV(sprs_vec);
            sgnl = obj.vid_region.comp1dTrnsfrm(sgnl, 3, @(X) mtrx_idct(X));
            sgnl = obj.vid_region.vectorize(sgnl);
        end

    end
    
end

