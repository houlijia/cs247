classdef CS_AnlsParams < ProcessingParams
    %CS_AnlsParams specifies parameters for analysis of CS measurements.
    
    properties
        % If true, all shift matches are made with the same set of
        % measurements. Otherwise, each comparison is made with all
        % measurements suitable for that particular offset.
        fxd_trgt = true;
        
        % Exponent of the norm used in comparisons (e.g. 1 for L1). -1
        % indicates using correlation coefficient.
        nrm_exp = 1;
        
        % If true run an analysis on reference input vectors as well.
        chk_ofsts=false;
        
        % The following parameters define the search area for checking
        % motion. The search is done in K stages. In each stage the search is
        % done in a rectangular grid centered at the optimal point of the
        % previous search ([0,0] in the first stage). The grid
        % resolution is given as a simple fraction. If not specified, the
        % grid resolution is 1 in the first stage and subsequently it is 
        % half the resoultion of the previous stage.   The
        % search in the k-th stage is done in all points (x,y) such that:
        %     (x,y)=(v+i*N(1)/D, h+j*N(2)/D)
        %     |x}|<=R(1,1),  |y|<=R(1,2)
        %     |x-v|<=R(k,1), |y-h|<=R(k,2)
        %     N is the numerator of the grid resolution in the k-th stage
        %       - row vector of size 2 - (vertical, horizontal)
        %     D is the denominator of the grid resolution in the k-th stage
        %     R specifies the range of search (in whole pixels) - array of
        %       size [K,2] where the k-th row is the [vertical,horizontal]
        %       range in the k-th stage. Note that R(1,:) is a global limit
        %       in all stages.
        %    If D and R are not sepcified in some stages, they are
        %    set to be half the value of the previous steps, with a value
        %    of 1 in the first step
        
        % the range R above
        m_range = [6 6];
        
        % R above
        m_step_numer = [1 1];
        
        % D above
        m_step_denom = [];
        
        % Number of pixels (vertical, horizontal) on each side of a given
        % pixel to average in edge detection
        edge_rng = [2,2];        
        
        % If true, blocks on edges will be ignored in selective decoding
        ignore_edge = true; 
        
        % Checking bacground options
        chk_bgrnd = struct(...
          'mx_avg', 10,... % Maximum number of blocks to average (0=no checking)
          'mn_dcd', 4,... % minimum number of blocks to make a decision
          'thrsh', 2 ... % Threshold for declaring bacground (in std.dev. units)
        );
    end
    
    methods
        function obj = CS_AnlsParams(def)
            if nargin > 0
                args = {def};
            else
                args = {};
            end
            obj = obj@ProcessingParams(args{:});
        end
        
        function setParams(obj, params)
            obj.setParams@ProcessingParams(params);
            bgn = size(obj.m_step_denom,1);
            len = size(obj.m_step_numer,1);
            if bgn < len
                obj.m_step_denom = [obj.m_step_denom; zeros(len-bgn,1)];
                for k=bgn+1:len
                    if k==1
                        obj.m_step_denom(k) = 1;
                    else
                        obj.m_step_denom(k) = obj.m_step_denom(k-1)*2;
                    end
                end
            end
            bgn = size(obj.m_step_numer,1);
            len = size(obj.m_range,1);
            if bgn < len
                obj.m_step_numer = [obj.m_step_numer(1:bgn,:); zeros(len-bgn,2)];
                obj.m_step_denom = [obj.m_step_denom(1:bgn,:); zeros(len-bgn,1)];
                for k=bgn+1:len
                    if k==1
                        obj.m_step_numer(k,:)=[1,1];
                        obj.m_step_denom(k) = 1;
                    elseif ~any(mod(obj.m_step_numer(k-1,:),2))
                        obj.m_step_numer(k,:) = obj.m_step_numer(k,:) /2;
                        obj.m_step_denom(k) = obj.m_step_denom(k-1);
                    else
                        obj.m_step_numer(k,:) = obj.m_step_numer(k-1,:);
                        obj.m_step_denom(k) = obj.m_step_denom(k-1)*2;
                    end
                end
            end
        end
            
        function str = classDescription(~)
          str = 'Analysis options';
        end

%         function str = describeParams(obj, prefix)
%             if nargin < 2
%                 prefix = '';
%             end
%             
%             str = sprintf('%s Measurementss analysis:\n%s',...
%               prefix, show_str(obj, '', ...
%               struct('prefix', prefix, 'struct_marked', true)));
%                        
%             function s = str_logical(x)
%                 if x
%                     s = 'true';
%                 else
%                     s = 'false';
%                 end
%             end
%             
%             str = sprintf('%s Measurements analysis:', prefix);
%             str = sprintf('%s\n%s    range=[%s], step=[%s]/[%s] edge_rng=[%s]', str, prefix,...
%                 show_str(obj.m_range, '%d'),...
%                 show_str(obj.m_step_numer, '%d'),...
%                 show_str(obj.m_step_denom, '%d'),...
%                 show_str(obj.edge_rng, '%d'));
%             str = sprintf('%s\n%s fxd_trgt=%s nrm_exp=%d chk_ofsts=%s ignore_edge=%s',...
%                 str, prefix, str_logical(obj.fxd_trgt), obj.nrm_exp, ...
%                 str_logical(obj.chk_ofsts), str_logical(obj.ignore_edge));
%         end
    end
    
end

