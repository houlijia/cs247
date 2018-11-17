classdef SqrErr < handle
    %SqrErr - an object keeping track of square error computation
    
    properties
        sum_sqr_err = 0;
        n_pnt = 0;
    end
    
    methods
        % number of argumens is 0 or 2
        function obj = SqrErr(ref,tst)
            if nargin > 0
                obj.update(ref,tst);
            end
        end
        
        function update(obj,ref,tst)
          if iscell(tst)
            for k=1:numel(tst)
              obj.update(ref{k}, tst{k});
            end
            return
          end
          
          df = double(tst(:)-ref(:));
          obj.sum_sqr_err = obj.sum_sqr_err + dot(df,df);
          obj.n_pnt = obj.n_pnt + length(df);
        end
        
        function mse = meanSqrErr(obj)
            mse = obj.sum_sqr_err / obj.n_pnt;
        end
        
        function [psnr, mse] = calcPSNR(obj, max_val)
            mse = obj.meanSqrErr();
            if mse
                psnr = 10*log10((max_val*max_val)/mse);
            else
                psnr = inf;
            end
        end
    end
    
    methods(Static)
        function [psnr, mse] = compPSNR(ref,tst,max_val)
            sqer = SqrErr(ref,tst);
            [psnr,mse] = sqer.calcPSNR(max_val);
        end
    end
    
end

