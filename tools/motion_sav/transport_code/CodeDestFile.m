classdef CodeDestFile < CodeDest
   % CodeDestFiel is an implementation of CodeDest as a disk file.
    
   properties (Constant)
        buf_size = 1024;
   end
   properties
        fh   % File handle
        fname   % file name
        total_bytes = 0;
        datalen = 0;
        buffer = zeros(CodeDestFile.buf_size,1);
        buf_len = 0;
    end
    
    methods
        % Constructor. if fdef is a character string it is interpreted as a
        % file name to open.  Otherwise it is assumed to be a file handle.
        function obj = CodeDestFile(fdef)
            if ischar(fdef)
                [obj.fh, emsg] = fopen(fdef, 'w');
                if obj.fh == -1
                    err = MException('CodeDestFile:OpenFailed',...
                        sprintf('failed opening file %s (%s)', ...
                        strrep(fdef,'\','\\'), emsg));
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
            if obj.buf_len > 0
                fwrite(obj.fh, obj.buffer(1:obj.buf_len), 'uint8');
            end
            if ischar(obj.fname)
                fclose(obj.fh);
            end
        end
 
        function err = write(obj, code)
            code_len = length(code);
            len = code_len + obj.buf_len;
            if len >= obj.buf_size
                cnt = fwrite(obj.fh, [obj.buffer(1:obj.buf_len); code(:)], 'uint8');
                if cnt ~= len
                    err = ferror(obj.fh);
                    return;
                else
                    err = 0;
                end
                obj.buffer = zeros(obj.buf_size,1);
                obj.buf_len = 0;
            else
                obj.buffer(obj.buf_len+1:obj.buf_len+code_len) = code(:);
                obj.buf_len = len;
                err = 0;
            end
            obj.datalen = obj.datalen + code_len;
            obj.total_bytes = obj.total_bytes + code_len;
        end
        
        function len = length(obj)
            len = obj.datalen;
        end
        
        function len = total(obj)
            len = obj.total_bytes;
        end
            
    end
    
end

