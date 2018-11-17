classdef CodeSourceFile < CodeSource
    % CodeSourceFile is an implementation of CodeSource as a disk file.
    
    properties
        fh   % File handle
        fname   % file name
        buffer = [];   % A buffer of pre-read content for efficiency.
        buf_size = 1024;  % Size of minimal read attempt
        buf_done = 0;         % Number bytes already read from the buffer.
    end
    
    methods
        % Constructor. if fdef is a character string it is interpreted as a
        % file name to open.  Otherwise it is assumed to be a file handle.
        function obj = CodeSourceFile(fdef)
            if ischar(fdef)
                [obj.fh, emsg] = fopen(fdef, 'r');
                if obj.fh == -1
                    err = MException('CodeSourceFile:OpenFailed',...
                        sprintf('failed opening file %s (%s)', ...
                        strrep(fdef, '\','\\'), emsg));
                    throw(err);
                else
                    obj.fname = fdef;
                end
            else
                obj.fh = fdef;
                obj.fname = -1;
            end
        end
        
        % Destructor. Close handle if it was opened by the constructor.
        function delete(obj)
            if ischar(obj.fname)
                fclose(obj.fh);
            end
        end
        
        function code = read(obj, cnt)
            cnt = double(cnt);
            code = [];
            avail = length(obj.buffer) - obj.buf_done;
            if avail >= cnt
                code = obj.buffer(obj.buf_done+1:obj.buf_done+cnt);
                obj.buf_done = obj.buf_done + cnt;
                return;
            elseif avail > 0
                code = uint8(zeros(cnt,1));
                code(1:avail) = obj.buffer(obj.buf_done+1:end);
                cnt = cnt - avail;
                obj.buffer = [];
                obj.buf_done = 0;
            end
            
            if cnt >= obj.buf_size
                if isempty(code)
                    [code, len] = fread(obj.fh, cnt, '*uint8');
                    if len < cnt
                        err = true;
                    else
                        return;
                    end
                else
                    [obj.buffer, len] = fread(obj.fh, cnt, '*uint8');
                    err = (len < cnt);
                end
            else
                [obj.buffer, len] = fread(obj.fh, obj.buf_size, '*uint8');
                 err = (len < cnt);
            end
            
            if err
                if feof(obj.fh)
                    if len == 0 && avail == 0
                        code = -1;
                    else
                        code = 'End of array deteced';
                    end
                else
                    code = ferror(obj.fh);
                end
                return;
            end
            
            code(avail+1:avail+cnt) = obj.buffer(1:cnt);
            if cnt == length(obj.buffer)
                obj.buffer = [];
            else
                obj.buf_done = cnt;
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
            max_cnt = double(max_cnt);
            nseq = double(nseq);
            cnt = 0;
            if ~nseq
                indcs = [];
                buf = uint8([]);
                return
            end
            
            ns = 0;
            indcs = zeros(1,nseq);
            offset = obj.buf_done;
            
            while ns<nseq
                eds = find(obj.buffer(offset+1:end)<128, nseq-ns);
                if ~isempty(eds)
                    cnt = cnt + eds(end);
                    if cnt > max_cnt
                        buf = 'Exceeded allowed number of bytes';
                        return
                    end
                    ns1 = ns+length(eds);
                    indcs(ns+1:ns1) = eds + (offset-obj.buf_done);
                    ns = ns1;
                    offset = offset + eds(end);
                    continue;
                else
                    % need to read
                    ns_left = nseq - ns;
                    nrd = max(min(4*ns_left, max_cnt-cnt), obj.buf_size);
                    [bfr, len] = fread(obj.fh, nrd, '*uint8');
                    obj.buffer = [obj.buffer; bfr];
                    if len < ns_left
                        if feof(obj.fh)
                            if length(obj.buffer) == obj.buf_done
                                buf = -1;
                            else
                                buf = 'End of file deteced';
                            end
                        else
                            buf = ferror(obj.fh);
                        end
                        return;
                    end
                end
            end
            buf = obj.buffer(obj.buf_done+1:obj.buf_done+indcs(end));
            obj.buf_done = obj.buf_done + indcs(end);
            if obj.buf_done == length(obj.buffer)
                obj.buffer = [];
                obj.buf_done = 0;
            elseif obj.buf_done > obj.buf_size
                obj.buffer = obj.buffer(obj.buf_done+1:end);
                obj.buf_done = 0;
            end
        end
    end
end

