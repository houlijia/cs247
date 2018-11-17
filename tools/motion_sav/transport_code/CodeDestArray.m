classdef CodeDestArray < CodeDest
    % CodeDestArray is an implementation of CodeDest as a memory array
    
    properties
        data        % Array of data
        total_bytes = 0;
        datalen = 0;
        incr_size = 1;
    end
    
    methods
        % CodeDestArray - constructor.
        function obj = CodeDestArray(incr_s)
            obj.data=uint8([]);
            if nargin > 0
                obj.incr_size = incr_s;
            end
        end
        
        function err = write(obj, code)
            codelen = length(code);
            newdatalen = obj.datalen + codelen;
            if obj.incr_size > 1
                datasize = length(obj.data);
                if newdatalen > datasize 
                    add_size = obj.incr_size * ...
                        ceil((newdatalen-datasize)/obj.incr_size);
                    obj.data(end+1:end+add_size)=zeros(1,add_size);
                end
            end
            obj.data(obj.datalen+1:newdatalen) = code;
            obj.datalen = newdatalen;
            obj.total_bytes = obj.total_bytes + codelen;
            err = 0;
        end
        
        function len = length(obj)
            len = obj.datalen;
        end
        
        function len = total(obj)
            len = obj.total_bytes;
        end
        
        % removeTail - remove the cnt oldest entries in the array. If cnt
        %              is not specified, remove all.
        function removeTail(obj, cnt)
            if nargin < 2 || cnt >= obj.datalen
                obj.datalen = 0;
            else
                obj.datalen = obj.datalen - cnt;
                obj.data(1:obj.datalen) = obj.data(cnt+1: cnt+obj.datalen);
            end
        end
    end
    
end

