classdef VidSparserTV < BaseSparser
  % VidSparserTV is a sparser class for video where the tranformation is Total
  % variation and the target function is the unisotropic measure, i.e. L1
  % norm of the horizontal and vertical differences.
  %   Detailed explanation goes here
  
  properties (Constant)
  end
  
  properties (SetAccess=protected, GetAccess=public)
    % way of computing differences with VidRegion.compDiff
    %            0 - inside differnces - the output dimension is
    %                smaller by one from the input dimentsion.
    %            1 - circular differences - the output dimension
    %                is the same as the input dimesnion.
    %            2 - extended - zero is added before the entries,
    %                so the first entry is preserved.
    mode=0;
    use_dc=false;
    ndim = 2;
    
    % pixelization parameters for vertical and horizontal differences
    diff_pxl_prms;
    
%     % Offset to beginning of k-th component
%     offset;
    
    % Time transform to be used by subclasses. Assumed to commute with the
    % TV sparsification operation
    t_trnsfrm=[];
    inv_t_trnsfrm = [];
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
    %              mode - Way of computing differences with VidRegion.compDiff
    %               0 - inside differnces - the output dimension is
    %                   smaller by one from the input dimentsion.
    %               1 - circular differences - the output dimension
    %                   is the same as the input dimesnion.
    %               2 - extended - zero is added before the entries,
    %                   so the first entry is preserved.
    %                default = 0 (obj.mode);
    %              use_dc - if true, the total variation matrix is
    %                        augmented by matrices of the DC value of each
    %                        color-block combination. Default=false
    %                        (obj.use_dc).
    function obj = VidSparserTV(args)
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
    %              mode - Way of computing differences with VidRegion.compDiff
    %                0 - inside differnces - the output dimension is
    %                   smaller by one from the input dimentsion.
    %                1 - circular differences - the output dimension
    %                   is the same as the input dimesnion.
    %                2 - extended - zero is added before the entries,
    %                   so the first entry is preserved.
    %                default = 0 (obj.mode);
    %              use_dc - if true, the total variation matrix is
    %                        augmented by matrices of the DC value of each
    %                        color-block combination. Default=false
    %                        (obj.use_dc).
    function set(obj, args)
      if isfield(args,'mode')
        obj.mode = args.mode;
      end
      if isfield(args,'use_dc')
        obj.use_dc = args.use_dc;
      end
      obj.set@BaseSparser(args);
    end
    
    function args = getArgs(obj) 
      args = obj.getArgs@BaseSparser();
      args.mode = obj.mode;
    end
    
    function setWgts(obj, wts)
      wgts_grp = cell(obj.ndim,1);
      wts = obj.vid_region.pixelize(wts, obj.blk_stt);
      for dim = 1:obj.ndim
        wtsd = VidRegion.getDiffPxls(wts, dim, obj.mode);
        wgts_grp{dim} = VidRegion.vectorize(wtsd);
      end
      obj.wgts = vertcat(wgts_grp{:});
    end    
  end
  
  methods (Access=protected)
    
    function setBaseMtx(obj, ~)
      prms = obj.vid_region.getParams_vecToBlks(obj.blk_stt);
      mtrx = cell(1, obj.ndim);
      
      if isempty(obj.t_trnsfrm)
        trnsfrm = SensingMatrixUnit(obj.expndr.nRows());
      else
        trnsfrm = VidRegion.get1dTrnsfrmMtrx(prms, 3, ...
          obj.t_trnsfrm, obj.inv_t_trnsfrm);
      end
      
      for dim=1:obj.ndim;
        mtrx{dim} = VidRegion.getDiffMtrx(prms, dim, obj.mode);
      end
      mtx_tv = SensingMatrixConcat.constructConcat(mtrx);
      mtx = SensingMatrixCascade.constructCascade({mtx_tv, trnsfrm});
      
      if obj.use_dc
        % Generate DC matrix
        n_c = size(prms.ofsts,1);
        n_b = size(prms.ofsts,2);
        if VidRegion.sameBlk(prms)
          mdc = SensingMatrixDC(prod(prms.blk_size(1,1:obj.ndim,1)));
          mdc.setNoClipFlag(false);
          scl = SensingMatrixScaler(...
            n_c*n_b*prod(prms.blk_size(1,obj.ndim+1:end,1)),...
            1/sqrt(n_c*n_b*prms.blk_len(1,1)));
          mtx_dc = SensingMatrixKron.constructKron({scl,mdc});
        else
          dc_mtrcs = cell(n_b, n_c);
          for iclr=1:n_c
            for iblk=1:n_b
              mdc = SensingMatrixDC(prod(prms.blk_size(iclr,1:obj.ndim,iblk)));
              mdc.setNoClipFlag(false);
              dc_mtrcs{iblk,iclr} = SensingMatrixKron.constructKron({...
                SensingMatrixUnit(prod(prms.blk_size(iclr,obj.ndim+1:end,iblk))),...
                mdc});
            end
          end
          scl = SensingMatrixScaler(...
            n_c*n_b*prod(prms.blk_size(1,obj.ndim+1:end,1)),...
            1/sqrt(sum(prms.blk_len(:))));
          mtx_dc = SensingMatrixCascade.constructCascade({scl, ...
            SensingMatrixBlkDiag(dc_mtrcs(:))});
        end
        
        obj.base_mtx = SensingMatrixConcat.constructConcat({mtx, mtx_dc});
      else
        obj.base_mtx = mtx;
      end
    end
    
  end
end

