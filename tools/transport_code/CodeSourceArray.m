classdef CodeSourceArray < CodeSource
    % CodeSource is an implementation of CodeSource in a memory array
    
    properties
        data    % Array of data
        n_read  % Number of bytes read
    end
    
    methods
        % Constructor.  arr is the input array (uint8).
        function obj=CodeSourceArray(arr)
            obj.data = arr;
            obj.n_read = 0;
        end
        
        % read function
        function code = read(obj, cnt)
            if cnt == 0
                code = uint8([]);
            elseif obj.n_read == length(obj.data)
                code = -1;
            elseif obj.n_read + cnt > length(obj.data)
                code = 'End of array deteced';
            else
                code = obj.data(obj.n_read+1:obj.n_read+cnt);
                obj.n_read = obj.n_read + cnt;
            end
        end
    end
    
    methods (Access=protected)
        % Read a sequence of bytes which contains a specified number of
        % bytes < 128 and return the seuqence and the indices.
        % Input:
        %    obj - this object
        %    max_cnt - (optional) maximal number of byte to read.
        %    nseq - number of bytes which are < 128
        % Output:
        %    buf - Output buffer (uint8)
        %          -1 if EOD was encountered before any byte was read. An 
        %          error string if an error occurred or if EOD was 
        %          encountered after some bytes were read or if max_cnt was 
        %          exceeded.
        %    indcs - indices of the bytes < 128 (row vector of of nseq entries)
        %    cnt - Number of bytes read.
        function [buf, indcs, cnt] = readSeqs(obj, max_cnt, nseq)
            cnt = 0;
            if ~nseq
                indcs = [];
                buf = uint8([]);
                return
            elseif obj.n_read == length(obj.data)
                buf = -1;
                return
            end
            
            indcs = find(obj.data(obj.n_read+1:end)<128, nseq);
            if length(indcs) < nseq
                buf = 'End of array deteced';
                return
            elseif  indcs(end) > max_cnt
                buf = 'Exceeded allowed number of bytes';
                return
            end
            cnt = indcs(end);
            buf = obj.data(obj.n_read+1:obj.n_read+cnt);
            obj.n_read = obj.n_read+cnt;
                
        end
    end
    
end

