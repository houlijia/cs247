classdef VidSparserTV < BaseSparser
    % VidSparserTV is a sparser class for video where the tranformation is Total
    % variation and the target function is the unisotropic measure, i.e. L1
    % norm of the horizontal and vertical differences.
    %   Detailed explanation goes here
    
    properties
        vid_region;  % Defines the structure of the video region
        n_sprsvec;   % dimension of the sparse vector
        n_sprsvec_v; % dimnsion of the differences along the vertical dimension
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
        function obj = VidSparserTV(varargin)
            obj.set(varargin{:});
        end

        function set(obj, varargin)
          varargin = parseInitArgs(varargin,{'vdrg', 'mode', 'expander'});
          if ~isempty(varargin) > 0
            vdrg = varargin{1};
            obj.set@BaseSparser(vdrg.vec_len);
            obj.vid_region = vdrg;
            if length(varargin) >= 2
              obj.mode = varargin{2};
            end
            
            % compute n_sprsvec
            obj.n_sprsvec_v = vdrg.compDiffLength(1,obj.mode);
            obj.n_sprsvec = obj.n_sprsvec_v + vdrg.compDiffLength(2,obj.mode);
            
            if length(varargin) > 2
                obj.setExpander(varargin{3});
            end
          end
        end
        
        % Returns the dimension of the sparse vector - 
        function n = dimSprsVec(obj)
            n = obj.n_sprsvec;
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
            if ~do_not_vectorize
                duv = vdrg_v.vectorize(duv);
                duh = vdrg_h.vectorize(duh);
            end
            sprs_vec = [duv; duh];
        end
    
        % Apply the transpose of the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        %  Output
        %     sgnl      - The input signal (of dimension n_sigvec)
        function sgnl = do_compSprsVecTrnsp(obj, sprs_vec)
            sv = obj.vid_region.compDiffTrnsp(sprs_vec(1:obj.n_sprsvec_v),1,...
                obj.mode);           
            sh = obj.vid_region.compDiffTrnsp(sprs_vec(obj.n_sprsvec_v+1:end),...
                2, obj.mode);
            sv = obj.vid_region.vectorize(sv);
            sh = obj.vid_region.vectorize(sh);
            sgnl = sv + sh;
        end
    end
end

