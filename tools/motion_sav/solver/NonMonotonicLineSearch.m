classdef NonMonotonicLineSearch < handle
    
    properties
        Q;
        C;
        cnt;
        cnt_out;
        alpha;
        alpha0;
        alpha_rate;
        decr_coef;
        wgt0;
        wgt_rate;
        wgt;
        failed_flag;
    end
    
    properties(Constant)
        cnt_max = 5;
        cnt_out_max = 0;
    end
    
    methods
        function obj = NonMonotonicLineSearch(opts)
            obj.decr_coef = opts.lsrch_c;
            obj.wgt0 = opts.lsrch_wgt;
            obj.wgt_rate = opts.lsrch_wgt_rate;
            obj.alpha_rate = opts.lsrch_alpha_rate;
        end
        
        function reset(obj, val)
            obj.Q = 1;
            obj.C = val;
            obj.wgt = obj.wgt0;
%            fprintf('reset(%8g):  C=%g Q=%g\n', val, obj.C, obj.Q);            
        end
        
        function start(obj, start_alpha)
            obj.cnt = 0;
            obj.cnt_out = 0;
            obj.failed_flag = false;
            if nargin < 2
                obj.alpha = 1;
            else
                obj.alpha = start_alpha;
            end
            obj.alpha0 = obj.alpha;
        end
        
        function result = failed(obj)
            result = obj.cnt >= obj.cnt_max && obj.cnt_out >=obj.cnt_out_max;
        end
        
        function give_up(obj, alpha_val)
            obj.failed_flag = true;
            obj.alpha = alpha_val;
            obj.wgt = obj.wgt * obj.wgt_rate;
        end
        
        function reduce_alpha(obj)
            if obj.cnt < obj.cnt_max;
                obj.alpha = obj.alpha * obj.alpha_rate;
                obj.cnt = obj.cnt + 1;
            else
                obj.alpha0 = obj.alpha0 /obj.alpha_rate;
                obj.alpha = obj.alpha0;
                obj.cnt = 0;
                obj.cnt_out = obj.cnt_out + 1;
            end
        end
        
        function ok = done_ok(obj)
            ok = (obj.cnt > 0 && ~obj.failed_flag);
        end
        
        function update(obj, val)
            Qp = obj.wgt*obj.Q;
            obj.Q =  Qp + 1;
            obj.C = (Qp*obj.C + val)/obj.Q;
        end
    end
    
end

