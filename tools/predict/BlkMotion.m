classdef BlkMotion < handle
    %BlkMotion - an object of this class represents the information
    %extracted from motion analysis in a block, based on prediction
    %analysis.  We assume that some array represented the effectiveness of
    %the prediction, with 1 being full prediction, 0 no prediction and
    %negative value indicates that prediction by past values makes things
    %worse.
    %   Detailed explanation goes here
    
    properties (Constant)
        csv_properties = { ...
            'mxv', 'mxp_v', 'mxp_h', 'mxp_t', ...
            'mdv', 'mdp_v', 'mdp_h', 'mdp_t',...
            'vlc_v', 'vlc_h';...
            'F','I','I','I', 'F','I','I','I', 'F', 'F'...
            };
        
        inf_val = -1000;
    end
    
    properties
        maxval = -inf; % maximum prediction value
        midval = -inf; % maximum prediciton when no temporal shift allowd
        maxpnt=[];     % Offset (v,h,t) at which maxval was attained.
        midpnt=[];     % Offset (v,h,t) at which midval was attained.
        vlcty=[];      % Speed of motion (v,h) in pixel/frame
        blk_info = []; % a struct describing the block, as returned by
                       % VidBlocker.getBlkInfo()
    end
    
    methods
        function obj = BlkMotion(mxval,mdval,mxpnt,mdpnt,vlct,b_info)
            if nargin > 1
                obj.setMotion(mxval, mdval, mxpnt, mdpnt, vlct);
                if nargin > 5
                    obj.blk_info = b_info;
                end
            end
        end
        
        function setMotion(obj, mxval, mdval, mxpnt, mdpnt, vlct)
            obj.maxval = max(mxval,obj.inf_val);
            obj.midval = max(mdval,obj.inf_val);
            obj.maxpnt = mxpnt;
            obj.midpnt = mdpnt;
            obj.vlcty = vlct;
        end
        
        function setBlkInfo(obj, blki)
            obj.blk_info = blki;
        end
        
        % Return 1 if motion found in a specific direction, -1 if 
        % if there are changes which cannot be explained by uniform motion,
        % and 0 otherwise
        function found = motionFound(obj)
          if obj.maxval <= 0
            found = -1;
          elseif obj.maxval > obj.midval
            found = 1;
          else
            found = 0;
          end
        end
        
        % Generate report string
        function str = report(obj)
            mxp = obj.maxpnt;
            mdp = obj.midpnt;
            mxp(1:2) = mxp(1:2) - mdp(1:2);
            mdp(1:2) = 0;
            vl = obj.vlcty;
            str = sprintf('Max: %f @(%d,%d,%d), %f @(%d,%d,%d) vl=(%f,%f)',...
                obj.maxval, mxp(1), mxp(2),mxp(3), obj.midval,...
                mdp(1), mdp(2), mdp(3), vl(1), vl(2));
        end
        
        % Return a struct with fields csv_properties
        function record = getCSVRecord(obj)
            record = obj.blk_info;
            
            record.mxv = obj.maxval;
            record.mxp_v = obj.maxpnt(1);                
            record.mxp_h = obj.maxpnt(2);                
            record.mxp_t = obj.maxpnt(3);                
            record.mdv  = obj.midval;
            record.mdp_v = obj.midpnt(1);                
            record.mdp_h = obj.midpnt(2);                
            record.mdp_t = obj.midpnt(3);                
            record.vlc_v = obj.vlcty(1);
            record.vlc_h = obj.vlcty(2);
        end
        
        function indx = getIndex(obj)
            indx = [obj.blk_info.indx_v, obj.blk_info.indx_h,...
                obj.blk_info.indx_t];
        end
    end
    
    methods (Static)
        function props = csvProperties()
            props1 = VidBlocker.getBlkInfoFields();
            props2 = BlkMotion.csv_properties;
            props = cell(2,size(props1,2)+size(props2,2));
            props(:,1:size(props1,2)) = props1;
            props(:,size(props1,2)+1:end) = props2;
        end
    
    end
    
end

