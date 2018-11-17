classdef VidSparserTV3 < VidSparserTV
    %VidSparserTV3 perform differences in 3 dimensions
    
    properties
        % combined length of differences along horizontal and vertical dimensions
        n_sprsvec_hv;  
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
        %      is_crcl (optional) if present perform cicrular TV
        function obj = VidSparserTV3(varargin)
          obj.set(varargin{:});
        end
        
        function set(obj, varargin)
          obj.set@VidSparserTV(varargin);
          obj.n_sprsvec_hv = obj.n_sprsvec;
          obj.n_sprsvec = obj.n_sprsvec_hv + vdrg.compDiffLength(3,obj.mode);
        end
            
        % Apply the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sgnl      - The input signal (of dimension n_sigvec)
        %  Output
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        function sprs_vec = do_compSprsVec(obj, sgnl, do_not_vectorize)
            if nargin < 3
                do_not_vectorize = false;
            end
            if ~iscell(sgnl)
                sgnl = obj.vid_region. pixelize(sgnl);
            end
            [duv, vdrg_v] = obj.vid_region.compDiff(sgnl,1, obj.mode);
            [duh, vdrg_h] = obj.vid_region.compDiff(sgnl,2, obj.mode);
            [dut, vdrg_t] = obj.vid_region.compDiff(sgnl,3, obj.mode);
            if ~do_not_vectorize
                duv = vdrg_v.vectorize(duv);
                duh = vdrg_h.vectorize(duh);
                dut = vdrg_t.vectorize(dut);
            end
            sprs_vec = [duv; duh; dut];
        end
    
        % Apply the transpose of the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        %  Output
        %     sgnl      - The input signal (of dimension n_sigvec)
        function sgnl = do_ompSprsVecTrnsp(obj, sprs_vec)
            sv = obj.vid_region.compDiffTrnsp(sprs_vec(1:obj.n_sprsvec_v), ...
                1, obj.mode);           
            sh = obj.vid_region.compDiffTrnsp(sprs_vec(...
                obj.n_sprsvec_v+1:obj.n_sprsvec_hv), 2, obj.mode); 
            st = obj.vid_region.compDiffTrnsp(sprs_vec(...
                obj.n_sprsvec_hv+1:end), 3, obj.mode); 
            sv = obj.vid_region.vectorize(sv);
            sh = obj.vid_region.vectorize(sh);
            st = obj.vid_region.vectorize(st);
            sgnl = sv + sh + st;
        end
   end
    
end

