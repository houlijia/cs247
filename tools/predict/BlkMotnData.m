classdef BlkMotnData < handle
  % BlkMotnData contains raw data about block motion. This data is used
  % to create BlkMotion ofbjects
  
  properties (SetAccess=immutable, GetAccess=public)
    % The following parameters specify the search and are set by the 
    % constructor. Search is done in L steps, or levels
    m_range; % Array of size [L,2] specifying the search ranges. the k-th row 
             % specifies the vertical and horizontal ranges at the k-th level.
    step;  % A cell array of L cells. The content of the k-th cell is a 
           % SimpleFractions of size [1,2] representing the vertical and 
           % horizontal step sizes.
    n_ofsts; % Total number of offsets in all levels
    srch_cnt; % no. of times to run search in each level\
    wait_first; % no. of itereation to wait before first search in each level
    wait_next; % no. of iteration to wait before subsequent searches in each level
  end
  
  properties (SetAccess=protected, GetAccess=public)
    % These properties are computed during the motion detection   
    offsets; % A cell array of SimpleFractions representing the search space
    mtch;    % A cell array of match values, corresponding to offsets
    mx;   % Maximum mtch value at each level
    mx_ind; % index of mx at each level
  end
  
  methods
    % Constructor. 
    %   Inputs arguments:
    %     opts (optional) Either CS_AnlsParams object or struct with the
    %     fields m_range, m_step_numer, m_step_denom, m_srch_cnt,
    %     m_wait_first, m_wait_next, with the same
    %     semantics as these properties in CS_Anls_Params
    function obj = BlkMotnData(opts)
      if nargin < 1
        opts = CS_AnlsParams();
      end
      if isempty(opts) || (isstruct(opts) && ~isfield(opts,'m_range'))
        obj.m_range = [];
        return
      end
      n_stages = size(opts.m_range,1);
      
      if isstruct(opts)&& ~isfield(opts,'m_step_denom')
          opts.m_step_denom = 1;
      end
      
      obj.m_range = opts.m_range;
      if isstruct(opts) && ~isfield(opts,'m_srch_cnt')
        obj.srch_cnt = ones(n_stages,1);
      else
        obj.srch_cnt = opts.m_srch_cnt;
      end
      if isstruct(opts) && ~isfield(opts,'m_wait_first')
        obj.wait_first = zeros(n_stages,1);
      else
        obj.wait_first = opts.m_wait_first;
      end
      if isstruct(opts) && ~isfield(opts,'m_wait_next')
        obj.wait_next = zeros(n_stages,1);
      else
        obj.wait_next = opts.m_wait_next;
      end
      obj.step = cell(n_stages,1);
      obj.offsets = cell(n_stages,1);
      obj.mtch = cell(n_stages,1);
      obj.mx = (-inf) * ones(n_stages,1);
      obj.mx_ind = zeros(n_stages,1);
      
      obj.n_ofsts = 0;
      for k=1:n_stages
        obj.step{k} = SimpleFractions(opts.m_step_numer(k,:),...
          opts.m_step_denom(k));
        obj.n_ofsts = obj.n_ofsts + ...
          prod(2*floor(obj.m_range(k,:)./obj.step{k})+1);
      end
    end
    
    % Returns a struct args which is suitable for building the same object
    function args = getArgs(obj) 
      n_stages = obj.nStages();
      m_step_numer = zeros(n_stages, 2);
      m_step_denom = zeros(n_stages, 1);
      for k=1:n_stages
        [m_step_numer(k,:), m_step_denom(k)] = rat(obj.step{k});
      end
      args = struct(...
        'm_range', obj.m_range,...
        'm_step_numer', m_step_numer,...
        'm_step_denom', m_step_denom,...
        'm_srch_cnt', obj.srch_cnt,...
        'm_wait_first', obj.wait_first,...
        'm_wait_next', obj.wait_next);
    end

    function val = nStages(obj)
      val = length(obj.m_range);
    end
    
    % Compute offsets for stage stg_num and save in obj.
    % orig is the origin - a SimpleFractions object of size [1 2]
    % representing (vertical,horizontal) position.
    function offsets = compOffsets(obj, stg_num, orig)
      m_rng1 = obj.m_range(1,:);
      m_rng = obj.m_range(stg_num,:);
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
        vlcty = (mxpnt(1:2) - mdpnt(1:2)) ./ mxpnt(3);
        
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
