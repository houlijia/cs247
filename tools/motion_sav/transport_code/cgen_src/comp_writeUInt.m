function [output, cnt, errString] = comp_writeUInt( val, nv )
%COMP_WRITEUINT Summary of this function goes here
%   Detailed explanation goes here
    errString ='';
            cnt = 0;
            if ~isempty(find(val<0,1))
                errString = 'Negative input to writeUInt';
                output=uint8(0);
                return;
            end                
            
            if islogical(val)
                val = uint64(val);
            elseif ~isinteger(val)
                val = round(val);
            end
            val = uint64(val);
            
            output = uint8(zeros(1,8*nv));
            for k=1:nv
                vl = val(k);
                % First generate an array representing the vector in a
                % little endian order, then reverse the array and write it
                % out
                code = uint8(zeros(1,nv));   % Initial allocation
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
            %err = obj.write(output(1:cnt));
end

