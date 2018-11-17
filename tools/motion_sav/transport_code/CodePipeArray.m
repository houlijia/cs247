classdef CodePipeArray < CodeDestArray & CodeSource
    % CodePipeArray Is both a code source and a code destination implmented
    % in a memory array
    
    properties
        n_read = 0;
        remove_tail_blk = 256;
    end
    
    methods
        function obj = CodePipeArray(incr_s, rmv_blk)
            if nargin > 0
                s_args = {incr_s};
            else
                s_args = {};
            end
            obj = obj@CodeDestArray(s_args{:});
            if nargin > 1
                obj.remove_tail_blk = rmv_blk;
            end
        end
        
        % read function (same as in CodeSourceArray).
        function code = read(obj, cnt)
            if cnt == 0
                code = uint8([]);
            elseif obj.n_read == obj.datalen
                code = -1;
            elseif obj.n_read + cnt > obj.datalen
                code = 'End of array deteced';
            else
                code = obj.data(obj.n_read+1:obj.n_read+cnt);
                obj.n_read = obj.n_read + cnt;
                if obj.n_read >= obj.remove_tail_blk || ...
                        obj.n_read == obj.datalen
                    obj.removeTail(obj.n_read);
                    obj.n_read = 0;
                end
            end
        end
        
    end
    
    methods (Access=protected)
        % Similar to the same function in CodeSourceArray:
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
        %    indcs - indices of the bytes < 128 row vector of (of nseq entries)
        %    cnt - Number of bytes read.
        function [buf, indcs, cnt] = readSeqs(obj, max_cnt, nseq)
          cnt = 0;
          if ~nseq
            indcs = [];
            buf = uint8([]);
            return
          elseif obj.n_read == obj.datalen
            indcs = [];
            buf = -1;
            return
          end
          
          indcs = find(obj.data(obj.n_read+1:obj.datalen)<128, nseq);
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

