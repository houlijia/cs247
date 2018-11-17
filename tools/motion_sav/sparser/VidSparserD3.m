classdef VidSparserD3 < BaseSparser
    %SparserD3 computes for each point, except along the edges 3
    %consecutive differences along vertical, horizontal and temporal
    %dimensions.
    
    properties
        vid_region;
        n_sprsvec;
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
        function obj = VidSparserD3(varargin)
            obj.set(varargin{:});
        end

        function set(obj, varargin)
          varargin = parseInitArgs(varargin,{'vdrg', 'mode', 'expander'});
          if ~isempty(varargin) > 0
            vdrg = varargin{1};
            obj.set@BaseSparser(vdrg.vec_len);
            if length(varargin) < 2
              md = obj.mode;
            else
              md = varargin{2};
              obj.mode = md;
            end
            obj.vid_region = cell(1,4);
            obj.vid_region{1} = vdrg;
            obj.vid_region{2} = obj.vid_region{1}.getDiffVidRegion(1, md);
            obj.vid_region{3} = obj.vid_region{2}.getDiffVidRegion(2, md);
            obj.vid_region{4} = obj.vid_region{3}.getDiffVidRegion(3, md);
            obj.n_sprsvec = obj.vid_region{4}.vec_len;
            
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
                sprs_vec = obj.vid_region{1}.pixelize(sgnl);
            else
                sprs_vec = sgnl;
            end
            sprs_vec = obj.vid_region{1}.compDiff(sprs_vec,1, obj.mode);
            sprs_vec = obj.vid_region{2}.compDiff(sprs_vec,2, obj.mode);
            sprs_vec = obj.vid_region{3}.compDiff(sprs_vec,3, obj.mode);
            if ~do_not_vectorize
                sprs_vec = obj.vid_region{4}.vectorize(sprs_vec);
            end            
%             % test correctness
%             nrm0 = dot(sprs_vec,sprs_vec);
%             if iscell(sgnl)
%                 sgnl = obj.vid_region{1}.vectorize(sgnl);
%             end
%             nrm1 = dot(sgnl, obj.compSprsVecTrnsp(sprs_vec));
%             fprintf('nrm0 = %f nrm1 = %f diff=%f\n',nrm0,nrm1,nrm0-nrm1);
        end

        % Apply the transpose of the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        %  Output
        %     sgnl      - The input signal (of dimension n_sigvec)
        function sgnl = do_compSprsVecTrnsp(obj, sprs_vec)
            if ~iscell(sprs_vec)
                sgnl = obj.vid_region{4}.pixelize(sprs_vec);
            else
                sgnl = sprs_vec;
            end
            
            sgnl = obj.vid_region{3}.compDiffTrnsp(sgnl,3, obj.mode);
            sgnl = obj.vid_region{2}.compDiffTrnsp(sgnl,2, obj.mode);
            sgnl = obj.vid_region{1}.compDiffTrnsp(sgnl,1, obj.mode);
            sgnl = obj.vid_region{1}.vectorize(sgnl);
        end
    end   
end

