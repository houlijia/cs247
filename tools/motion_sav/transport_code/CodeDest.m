classdef CodeDest < matlab.mixin.Copyable
    % CodeDest is an abstract class representing an object into which code
    % can be written.
    % The destination can be a memory array, a file, a socket, etc.  The only
    % requirement is that it has to support sequential writing.
    % In the code, unsigned integer has a variable size representation. The
    % unsigned integer is represented as a sequence of bytes in big endian
    % (more significant bytes first) form. In each bytes the lower 7 bits
    % contain value while the MSB is a continuation bit (1 for non-final
    % byte 0 for final byte).
    % Signed integers are also supported.  The code for a signed integer is in
    % a sign-magnitude form and similar to the code for unsigned integers,
    % except that the 2nd MSB of the first byte (bit 6) is the
    % sign bit - 1 for negative and 0 for non-negative.
    % Code is usually stored in a length-value format, meaning an unsigned
    % integer in the above representation indicating the code length
    % followed by a sequence of bytes of the specified length.  Usually the
    % length is preceded by a type (key) indicating a code of what it is. 
    
    properties
    end
    
    methods (Abstract)
        % write - write an array bytes.
        % Input:
        %   obj - this object
        %   code - an array of uint8 (the number of bytes is the length of
        %          the array).  
        % Output:
        %   err is 0 for success or an error string if an error occurred.
        err = write(obj, code)
        
        % length - amount of bytes in  storage
        len = length(obj)
        
        % total - total amount of bytes that have been written into this
        %         object (not cleared by operations like rewind)
        len = total(obj)
    end
    
    methods
        % writeUInt - write an unsigned integer
        % Input:
        %   obj - this object
        %   val: unsigned integer value or vector.  
        %        Should not exceed UInt64 range.
        % Output:
        %   cnt - If successful, number of bytes used to write the integer,
        %         else an error string.
        function cnt = writeUInt(obj, val)
            if ~isempty(find(val<0,1))
                cnt = 'Negative input to writeUInt';
                return;
            end                
            val1=val;
            
            if islogical(val)
                val = uint64(val);
            elseif ~isinteger(val)
                val = uint64(round(val));
            end
            
            cnt = 0;
            output = uint8(zeros(1,8*numel(val)));
            for k=1:numel(val)
                vl = val(k);
                % First generate an array representing the vector in a
                % little endian order, then reverse the array and write it
                % out
                code = uint8(zeros(1,8));   % Initial allocation
                n = 1;
                while vl >=  128
                    code(n) = 128 + bitand(vl,127);
                    vl = bitshift(vl,-7);
                    n = n+1;
                end
                if n > 1
                    code(n) = 128 + vl;
                    code(1) = code(1) - 128;
                    code = fliplr(code(1:n));
                else
                    code(1) = vl;
                end
                
                output(cnt+1:cnt+n) = code(1:n);
                cnt = cnt + n;
            end
            [out1,cnt1,err1] = comp_writeUInt(val1(:), numel(val1));
            %[out1,cnt1,err1] = comp_writeUInt_mex(val1(:), uint32(numel(val1)));
            if ~isempty(err1)
                cnt1 = err1;
            end
            if ~isequal(out1,output) || ~isequal(cnt1,cnt)
                error('mismatch');
            end
            err = obj.write(output(1:cnt));
            if ischar(err)
                cnt = err;
                return
            end
        end
        
        % writeSInt - write an integer
        % Input:
        %   obj - this object
        %   val: An signed numeric value or vector
        % Output:
        %   cnt - If successful, number of bytes used to write the integer,
        %         else an error string.
        function cnt = writeSInt(obj, val)
            if ~isinteger(val)
                val = int64(round(val));
            end
            
            cnt = 0;
            output = uint8(zeros(1,8*numel(val)));
            for k=1:numel(val)
                vl = val(k);
                if vl < 0
                    ngtv = true;
                    vl = uint64(-vl);
                else
                    vl = uint64(vl);
                    ngtv = false;
                end
                % First generate an array representing the vector in a
                % little endian order, then reverse the array and write it
                % out
                code = uint8(zeros(1,8));   % Initial allocation
                n = 1;
                while vl >= 64
                    code(n) = 128 + bitand(vl,127);
                    vl = bitshift(vl,-7);
                    n = n+1;
                end
                if ngtv
                    vl = vl + 64;
                end
                if n > 1
                    code(n) = 128 + vl;
                    code(1) = code(1) - 128;
                    code = fliplr(code(1:n));
                else
                    code(1) = vl;
                end
                
                output(cnt+1:cnt+n) = code(1:n);
                cnt = cnt + n;
            end
            err = obj.write(output(1:cnt));
            if ischar(err)
                cnt = err;
                return
            end
        end
        
        % writeNumber - write a number or vector(double)
        % Input:
        %   obj - this object
        %   val: A value or a vector of values
        % Output:
        %   cnt - If successful, number of bytes used to write the integer,
        %         else an error string.
        function cnt = writeNumber(obj, val)
            e_min = -1022;
            [f,e] = log2(val);
            for k=1:numel(val)
                f0 = f(k);  e0 = e(k);
                                
                % [f0,e0] is a mantissa-exponent pair representing val(k).
                % We try to find an equivalent pair [f1,e1] such that
                % f1 is integer. If we cannot find such a pair we let [f1,e1]
                % be such that f1 e1 = e_min.
                f1 = f0; e1 = e0;
                m = 0;
                while f1 > floor(f1)
                    if e1 <= e_min
                        break;
                    end
                    m = 2*m + 1;
                    if m > e1 - e_min
                        m = e1 - e_min;
                    end
                    f1 = pow2(f1,m);
                    e1 = e1 - m;
                    if f1 == floor(f1)
                        break % f1 is integer
                    else
                        f0 = f1; e0 = e1;
                    end
                end
                
                while e0 - e1 > 1
                    m = floor((e0 - e1)/2);
                    ff = pow2(f0,m);
                    ee = e0-m;
                    if ff > floor(ff)
                        f0 = ff; e0 = ee;
                    else
                        f1 = ff; e1 = ee;
                    end
                end
                f(k) = floor(f1);
                e(k) = e1;
            end
            cnt = obj.writeSInt(f);
            if ischar(cnt)
                return;
            end
            n = obj.writeSInt(e);
            if ischar(n)
                cnt = n;
                return;
            end
            cnt = cnt + n;
        end
        
        % writeBits - write an array of binary values as a series of bytes.
        % Input
        %   obj - this object.
        %   bitvals - an array of values, where non-zero is interpreted as a 1 bit
        %   and zero a 0 bit.
        % Output
        %   cnt - normally the number of bytes written. An error string if anerror
        %   occurred.
        function cnt = writeBits(obj, bitvals)
            len_b = length(bitvals);
            cnt = obj.writeUInt(len_b);
            if ischar(cnt) || len_b == 0; return; end
            
            remainder = rem(len_b, 8);
            b_array = uint8(zeros(1,ceil(len_b/8)));
            if remainder
                b_cnt = remainder;
            else
                b_cnt = 8;
            end
            
            k_array = 1;
            k_val = 1;
            while k_val <= len_b
                e_val = k_val + b_cnt;
                bt = uint8(0);
                for j=k_val:e_val-1
                    bt = bt*2;
                    if bitvals(j); bt = bt + 1; end
                end
                b_array(k_array) = bt;
                k_array = k_array+1;
                k_val = e_val;
                b_cnt = 8;
            end
            
            err = obj.write(b_array);
            if ischar(err)
                cnt = err;
            else
                cnt = cnt + length(b_array);
            end
        end
        
        % writeCode - write a length value code. First write the code length
        % and then writes the byte array of that length.
        % Input:
        %   code - uint8 vector to be written.  The length is the length
        %   of the vector.
        % Output:
        %   cnt - If successful, number of total bytes written, otherwise,
        %   an error code.
        function cnt = writeCode(obj, code)
            len_cnt = obj.writeUInt(length(code));
            if ischar(len_cnt)
                cnt = len_cnt;
            else
                err = obj.write(code);
                if ~ischar(err)
                    cnt = length(code) + len_cnt;
                end
            end
        end
        
        %writeString - writes a string
        function cnt = writeString(obj, str)
            data = uint8(str);
            cnt = obj.writeCode(data);
        end
    end
    
end

