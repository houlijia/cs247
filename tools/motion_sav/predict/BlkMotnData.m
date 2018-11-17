classdef BlkMotnData < handle
  % BlkMotnData contains raw data about block motion. This data is used
  % to create BlkMotion ofbjects
  
  properties
    m_range; % Array of search ranges, copied from opts.
    step;  % SimpleFractions representing the step sizes
    offsets; % A cell array of SimpleFractions representing the search space
    mtch;    % A cell array of match values, corresponding to offsets
    mx;   % Maximum mtch value at each level
    mx_ind; % index of mx at each level
  end
  
  methods
    % Constructor. opts is a CS_AnlsParams object
    function obj = BlkMotnData(opts)
      if nargin < 1
        opts = CS_AnlsParams();
      end
      n_stages = size(opts.m_range,1);
      obj. m_range = opts.m_range;
      obj.step = cell(n_stages,1);
      obj.offsets = cell(n_stages,1);
      obj.mtch = cell(n_stages,1);
      obj.mx = (-inf) * ones(n_stages,1);
      obj.mx_ind = zeros(n_stages,1);
      
      for k=1:n_stages
        obj.step{k} = SimpleFractions(opts.m_step_numer(k,:),...
          opts.m_step_denom(k));
      end
    end

    function val = nStages(obj)
      val = length(obj.m_range);
    end
    
    % Compute offsets for stage stg_num and save in obj.
    % orig is the origin - a SimpleFractions object of size [1 2]
    % representing (vertical,horizontal) position.
    function offsets = compOffsets(obj, stg_num, orig)
      m_rng1 = obj.m_range(1,:);
      m_rng = obj.m_range(stg_num);
      offsets = next_frm_corr_ofsts(orig, obj.step{stg_num},...
        max(-m_rng1, orig-m_rng), min(m_rng1, orig+m_rng));
      obj.offsets{stg_num} = offsets;
    end
    
    function stage_data = getStageData(obj, stg_num)
      stage_data = struct(...
        'step',obj.step{stg_num},...
        'offsets', obj.offsets{stg_num},...
        'mtch', obj.mtch{stg_num},...
        'mx', obj.mx(stg_num),...
        'mx_ind', obj.mx_ind(stg_num));
    end
    
    function obj = setStageData(obj, stg_num, stage_data)
      flds = fieldnames(stage_data);
      for k=1:length(flds)
        fld = flds{k};
        if iscell(obj.(fld))
          obj.(fld){stg_num} = stage_data.(fld);
        else
          obj.(fld)(stg_num) = stage_data.(fld);
        end
      end
    end
    
    % Compute a BlkMotion object
    function blk_motion = next_frm_corr_sort(obj)
      lvl = obj.nStages();
      
      % Search for best level
      xc = obj.getStageData(lvl);
      while lvl > 1
        if isempty(xc.mx_ind)
          lvl = lvl-1;
          xc = obj.getStageData(lvl);
          continue;
        end
        prev_lvl = lvl;
        
        for k=1:lvl-1
          xprev = obj.getStageData(k);
          if isempty(xprev.mx_ind)
            continue;
          end
          for j=1:size(xprev.offsets,1)
            if all(xc.offsets(xc.mx_ind,:) == xprev.offsets(j,:))
              lvl = k;
              break;
            end
          end
          if lvl ~= prev_lvl
            xc = xprev;
            break;
          end
        end
        if lvl == prev_lvl
          break;
        end
      end
      
      if ~isempty(xc.mx_ind)
        mxpnt = rat(xc.offsets(xc.mx_ind,:));
        mdpnt = rat(xc.offsets(1,:));
        vlcty = (mdpnt(1:2) - mxpnt(1:2)) ./ mxpnt(3);
        
        blk_motion = BlkMotion(xc.mx, xc.mtch(1), mxpnt, mdpnt, vlcty);
      else
        blk_motion = BlkMotion();
      end
      
    end
    
  end
  
  methods (Static)
    function max_info = compMax(mtch)
      [x, x_ind] = max(mtch);
      max_info = struct('mtch', mtch, 'mx', x, 'mx_ind', x_ind);
    end
  end
end
