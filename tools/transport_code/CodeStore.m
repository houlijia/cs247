classdef CodeStore < matlab.mixin.Copyable
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
  
  properties (Constant)
    % Minimum  number of elements in GPU array to justify encoding in GPU.
    gpu_min_usage =16;
  end
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
    
  end
  
  methods
    
    function obj = CodeStore()
    end
  end
  
  methods (Static)
    function vlen = encodeLengthUInt(vl, use_gpu)
      % encodeLengthUInt computes how many bytes are necessary to compute an
      % unsigned integer array vl. The output is vlen contains the number of
      % bytes for each member of vl.
      if nargin < 2
        use_gpu = false;
      end
      
      if ~use_gpu
        vlen = ones(size(vl));
      else
        vlen = gpuArray.ones(size(vl));
      end
      thresh = 128;
      indcs = find(vl >= thresh);
      while ~isempty(indcs)
        sbst = vl(indcs);
        vlen(indcs) = vlen(indcs) + 1;
        thresh = thresh*128;
        indcs = indcs(sbst >= thresh);
      end
    end
    
    function len = encodeLengthBits(bitvals, n_bits)
      len = CodeStore.encodeLengthUInt(length(bitvals));
      if nargin < 2
        len = len + CodeStore.encodeLengthBitsArray(bitvals);
      else
        len = len + CodeStore.encodeLengthBitsArray(bitvals, n_bits);
      end
    end
    
    function len = encodeLengthBitsArray(bitvals, n_bits)
      if nargin < 2
        n_bits = 1;
      end
      len = ceil(length(bitvals)*n_bits/8);
    end


  end
  
  methods
    function output=encodeUInt(val)
      
      if isempty(val)
        output = uint8(zeros(1,0));
      else
        if isa(val, 'gpuArray')
          if numel(val) >= obj.gpu_min_usage
            use_gpu = true;
            vl = val(:)';
          else
            use_gpu = false;
            vl = gather(val(:)');
          end
        else
          use_gpu = false;
          vl = val(:)';
        end
        
        vlen =  CodeStore(vl, use_gpu);
        vend = cumsum(vlen);
        if ~use_gpu
          output = uint8(128*ones(1,vend(end)));
        else
          output = uint8(128*gpuArray.ones(1,vend(end)));
        end
        
        vr = rem(vl,128);
        output(vend) = uint8(vr);  % last byte. Clears continuation flag
        vl = (vl - vr)/128;
        k=1;
        indcs = find(vl);
        vleft = vl(indcs);
        while ~isempty(vleft)
          vr = rem(vleft,128);
          out_indcs = vend(indcs)-k;
          output(out_indcs) = output(out_indcs) + uint8(vr);
          vleft = (vleft-vr)/128;
          k=k+1;
          nz_indcs = find(vleft ~= 0);
          indcs = indcs(nz_indcs);
          vleft = vleft(nz_indcs);
        end
        if use_gpu
          output = gather(output);
        end
      end
      %         ref_output = output;
      %
      %         if ~isa(val,'gpuArray')
      %           cnt = 0;
      %           output = uint8(zeros(1,8*numel(val)));
      %           if islogical(val)
      %             val = uint64(val);
      %           elseif ~isinteger(val)
      %             val = uint64(round(val));
      %           end
      %
      %           for k=1:numel(val)
      %             vl = val(k);
      %             % First generate an array representing the vector in a
      %             % little endian order, then reverse the array and write it
      %             % out
      %             code = uint8(zeros(1,8));   % Initial allocation
      %             n = 1;
      %             while vl >=  128
      %               code(n) = 128 + bitand(vl,127);
      %               vl = bitshift(vl,-7);
      %               n = n+1;
      %             end
      %             if n > 1
      %               code(n) = 128 + vl;
      %               code(1) = code(1) - 128;
      %               code = fliplr(code(1:n));
      %             else
      %               code(1) = vl;
      %             end
      % %             ref_len = CodeDest.lengthUInt(val(k));
      % %             if n ~= ref_len
      % %               error('Wrong length');
      % %             end
      %             output(cnt+1:cnt+n) = code(1:n);
      %             cnt = cnt + n;
      %           end
      %           output = output(1:cnt);
      %         else
      % %           out_ref = encodeUInt(gather(val));
      %
      %           cnt = gpuArray(0);
      %           output = gpuArray(uint8(zeros(1,8*numel(val))));
      %           for k=1:numel(val)
      %             vl = val(k);
      %             % First generate an array representing the vector in a
      %             % little endian order, then reverse the array and write it
      %             % out
      %             code = gpuArray(uint8(zeros(1,8)));   % Initial allocation
      %             n = 1;
      %             while vl >=  128
      %               code(n) = 128 + rem(vl,128);
      %               vl = fix(vl/128);
      %               n = n+1;
      %             end
      %             if n > 1
      %               code(n) = 128 + vl;
      %               code(1) = code(1) - 128;
      %               code = fliplr(code(1:n));
      %             else
      %               code(1) = vl;
      %             end
      %
      %             output(cnt+1:cnt+n) = code(1:n);
      %             cnt = cnt + n;
      %           end
      %           output = gather(output(1:cnt));
      % %           if ~isequal(out_ref, output)
      % %             error('GPU encode failed');
      % %           end
      %         end
      %
      %         if ~isequal(ref_output,output)
      %           error('not matching')
      %         end
      
    end
    
    function cnt = writeUInt(obj, val)
      % writeUInt - write an unsigned integer
      % Input:
      %   obj - this object
      %   val: unsigned integer value or vector.
      %        Should not exceed UInt64 range.
      % Output:
      %   cnt - If successful, number of bytes used to write the integer,
      %         else an error string.
      if ~isempty(find(val<0,1))
        cnt = 'Negative input to writeUInt';
        return;
      end
      
      out = encodeUInt_mex(val);
      %       ref_out = encodeUInt(val);
      %       if ~isequal(out, ref_out)
      %         error('mex_output and output are not equal')
      %       end
      err = obj.write(out);
      if ischar(err)
        cnt = err;
      else
        cnt = length(out);
      end
    end
    
    function output = encodeSInt(val)
      if isempty(val)
        output = uint8(zeros(1,0));
      else
        if isa(val, 'gpuArray')
          if false && numel(val) >= obj.gpu_min_usage
            use_gpu = true;
            vl = val(:)';
          else
            use_gpu = false;
            vl = gather(val(:)');
          end
        else
          use_gpu = false;
          vl = val(:)';
        end
        neg_indcs = find(vl<0);
        vl(neg_indcs) = -vl(neg_indcs);
        
        if ~use_gpu
          vlen = ones(size(vl), 'uint32');
          incrmnt = ones(1,1,'uint32');
          thresh = 64;
        else
          vlen = gpuArray.ones(size(vl), 'uint32');
          incrmnt = gpuArray.ones(1,1,'uint32');
          thresh = gpuArray(64);
        end
        indcs = find(vl >= thresh);
        while ~isempty(indcs)
          sbst = vl(indcs);
          vlen(indcs) = vlen(indcs) + incrmnt;
          thresh = thresh*128;
          indcs = indcs(sbst >= thresh);
        end
        
        %           if isinteger(vl)
        %             vln = (nextpow2((vl+32)/64)+6+4)/7;
        %           else
        %             vln = 1+fix((nextpow2(fix(vl/64+1))+6)/7);
        %           end
        %           if ~isequal(vlen,vln)
        %             error('vlen~= vln')
        %           end
        
        vend = cumsum(vlen);
        if ~use_gpu
          output = uint8(128*ones(1,vend(end)));
        else
          output = uint8(128*gpuArray.ones(1,vend(end)));
        end
        
        vr = rem(vl,128);
        output(vend) = uint8(vr);  % last byte. Clears coninuation flag
        vl = (vl - vr)/128;
        k=1;
        indcs = find(vl);
        vleft = vl(indcs);
        while ~isempty(vleft)
          vr = rem(vleft,128);
          out_indcs = vend(indcs)-k;
          output(out_indcs) = output(out_indcs) + uint8(vr);
          vleft = (vleft-vr)/128;
          k=k+1;
          nz_indcs = find(vleft ~= 0);
          indcs = indcs(nz_indcs);
          vleft = vleft(nz_indcs);
        end
        
        neg_flg_indcs = vend(neg_indcs) - vlen(neg_indcs) + 1;
        output(neg_flg_indcs) = output(neg_flg_indcs) + 64;
        
        if use_gpu
          output = gather(output);
        end
      end
      %         ref_output = output;
      %
      %         if ~isa(val,'gpuArray')
      %           cnt = 0;
      %           output = uint8(zeros(1,8*numel(val)));
      %           if ~isinteger(val)
      %             val = int64(round(val));
      %           end
      %
      %           for k=1:numel(val)
      %             vl = val(k);
      %             if vl < 0
      %               ngtv = true;
      %               vl = uint64(-vl);
      %             else
      %               vl = uint64(vl);
      %               ngtv = false;
      %             end
      %             % First generate an array representing the vector in a
      %             % little endian order, then reverse the array and write it
      %             % out
      %             code = uint8(zeros(1,8));   % Initial allocation
      %             n = 1;
      %             while vl >= 64
      %               code(n) = 128 + bitand(vl,127);
      %               vl = bitshift(vl,-7);
      %               n = n+1;
      %             end
      %             if ngtv
      %               vl = vl + 64;
      %             end
      %             if n > 1
      %               code(n) = 128 + vl;
      %               code(1) = code(1) - 128;
      %               code = fliplr(code(1:n));
      %             else
      %               code(1) = vl;
      %             end
      %
      % %             ref_len = CodeDest.lengthSInt(val(k));
      % %             if n ~= ref_len
      % %               error('Wrong length');
      % %             end
      %             output(cnt+1:cnt+n) = code(1:n);
      %             cnt = cnt + n;
      %           end
      %           output = output(1:cnt);
      %         else
      % %           out_ref = encodeSInt(gather(val));
      %
      %           cnt = gpuArray(0);
      %           output = gpuArray(uint8(zeros(1,8*numel(val))));
      %           for k=1:numel(val)
      %             vl = val(k);
      %             if vl < 0
      %               ngtv = true;
      %               vl = -vl;
      %             else
      %               ngtv = false;
      %             end
      %             % First generate an array representing the vector in a
      %             % little endian order, then reverse the array and write it
      %             % out
      %             code = gpuArray(uint8(zeros(1,8)));   % Initial allocation
      %             n = 1;
      %             while vl >= 64
      %               code(n) = 128 + rem(vl,128);
      %               vl = fix(vl/128);
      %               n = n+1;
      %             end
      %             if ngtv
      %               vl = vl + 64;
      %             end
      %             if n > 1
      %               code(n) = 128 + vl;
      %               code(1) = code(1) - 128;
      %               code = fliplr(code(1:n));
      %             else
      %               code(1) = vl;
      %             end
      %
      %             output(cnt+1:cnt+n) = code(1:n);
      %             cnt = cnt + n;
      %           end
      %           output = gather(output(1:cnt));
      % %           if ~isequal(out_ref, output)
      % %             error('GPU encode failed');
      % %           end
      %         end
      %
      %         if ~isequal(ref_output,output)
      %           error('not matching')
      %         end
    end
    
    function cnt = writeSInt(obj, val)
      % writeSInt - write an integer
      % Input:
      %   obj - this object
      %   val: An signed numeric value or vector
      % Output:
      %   cnt - If successful, number of bytes used to write the integer,
      %         else an error string.
      
      
      out = encodeSInt_mex(val);
%       ref_out = encodeSInt(val);
%       if ~isequal(out, ref_out)
%         error('mex_output and output are not equal')
%       end
      err = obj.write(out);
      if ischar(err)
        cnt = err;
      else
        cnt = length(out);
      end
    end
    
    function cnt = writeSIntOffset(obj, val, ofst)
      % Same as writeSInt, but subtracting ofst (a scalar) from val before
      % writing out
      
      out = encodeSInt_mex(val, ofst);
      
%       ref_out = encodeSInt_mex(val-ofst);
%       if ~isequal(ref_out, out)
%         error('encodeSInt_mex incorrect with offset');
%       end

      err = obj.write(out);
      if ischar(err)
        cnt = err;
      else
        cnt = length(out);
      end
    end
    
    function cnt = writeNumber(obj, val)
      % writeNumber - write a number or vector(double)
      % Input:
      %   obj - this object
      %   val: A value or a vector of values
      % Output:
      %   cnt - If successful, number of bytes used to write the integer,
      %         else an error string.
      out = encodeNumber_mex(val);
      err = obj.write(out);
      if ischar(err)
        cnt = err;
      else
        cnt = length(out);
      end
    
%       if ~isa(val,'gpuArray')
%         [f,e] = obj.writeNumberCPU(val);
%       else
%         [f,e] = obj.writeNumberCPU(gather(val));
% %         [f,e] = obj.writeNumberGPU(val);
%       end
%       
%       cnt = obj.writeSInt(f);
%       if ischar(cnt)
%         return;
%       end
%       n = obj.writeSInt(e);
%       if ischar(n)
%         cnt = n;
%         return;
%       end
%       cnt = cnt + n;
    end
    
    function cnt = writeBits(obj, bitvals, n_bits)
      % writeBits - write an array of binary values as a series of bytes,
      % including the length of the array
      % Input
      %   obj - this object.
      %   bitvals - an array of values, where non-zero is interpreted as a 1 bit
      %   and zero a 0 bit.
      %   n_bits - (optional) if present bitvals is assumed to be
      %     integers of n_bits bits.
      % Output
      %   cnt - normally the number of bytes written. An error string if anerror
      %   occurred.
      len_b = length(bitvals);
      cnt = obj.writeUInt(len_b);
      if ischar(cnt) || len_b == 0; return; end
      
      if nargin < 3
        len = obj.writeBitsArray(bitvals);
      else
        len = obj.writeBitsArray(bitvals, n_bits);
      end
      if ischar(len)
        cnt = len;
      else
        cnt = cnt + len;
      end
      
    end
    
    function cnt = writeBitsArray(obj, bitvals, n_bits)
      % writeBits - write an array of binary values as a series of bytes,
      % NOT including the length of the array
      % Input
      %   obj - this object.
      %   bitvals - an array of values, where non-zero is interpreted as a 1 bit
      %     and zero a 0 bit.
      %   n_bits - (optional) if present bitvals is assumed to be
      %     integers of n_bits bits.
      % Output
      %   cnt - normally the number of bytes written. An error string if anerror
      %   occurred.
      len_b = numel(bitvals);
      
      if nargin >= 3
        bitv = zeros(len_b, n_bits, 'uint8');
        cpu_bitvals = gather(bitvals);
        for k=1:n_bits
          bitv(:,k) = bitand(bitshift(cpu_bitvals,-(k-1)),1);
        end
        cnt = obj.writeBitsArray(bitv(:));
        return
      end
      
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
        cnt = length(b_array);
      end
    end
    
    function cnt = writeCode(obj, code)
      % writeCode - write a length value code. First write the code length
      % and then writes the byte array of that length.
      % Input:
      %   code - uint8 vector to be written.  The length is the length
      %   of the vector.
      % Output:
      %   cnt - If successful, number of total bytes written, otherwise,
      %   an error code.
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
    
    function cnt = writeString(obj, str)
      %writeString - writes a string
      data = uint8(str);
      cnt = obj.writeCode(data);
    end
  end
  
  methods(Static)
    function [f,e] = integerize(f,e)
      if f==0
        return
      end
      
      e_min = -1022;
      f1 = f;  e1 = e;
      m=3;  % Initial guess for step in e
      
      % [f0,e0] is a mantissa-exponent pair representing val(k) in 
      % which f0 is not an integer.
      % We try to find an equivalent pair [f1,e1] such that
      % f1 is integer. If we cannot find such a pair we let [f1,e1]
      % be such that f1 e1 = e_min. The loop will occur at least once,
      % Since the mantissa in the output of log2() has magnitude < 1.
      while f1 > floor(f1) && e1> e_min
        f0 = f1; e0 = e1;
        m = min(2*m + 1, e1-e_min);
        f1 = pow2(f1,m);
        e1 = e1 - m;
      end
      
      while m > 1
        m1 = floor(m/2);
        ff = pow2(f0,m1);
        ee = e0-m1;
        if ff > floor(ff)
          f0 = ff; e0 = ee;
          m = m - m1;
        else
          f1 = ff; e1 = ee;
          m = m1;
        end
      end
      f = floor(f1);
      e = e1;
    end
  
    function [f,e] = writeNumberCPU(val)
      [f,e] = log2(val);
      
      [f,e] = arrayfun(@CodeDest.integerize, f,e);

%       e_min = -1022;
%       
%       [f_ref,e_ref] = arrayfun(@CodeDest.integerize, f,e);
% 
%       for k=1:numel(val)
%         f0 = f(k);  e0 = e(k);
%         
%         % [f0,e0] is a mantissa-exponent pair representing val(k).
%         % We try to find an equivalent pair [f1,e1] such that
%         % f1 is integer. If we cannot find such a pair we let [f1,e1]
%         % be such that f1 e1 = e_min.
%         f1 = f0; e1 = e0;
%         m = 0;
%         while f1 > floor(f1)
%           if e1 <= e_min
%             break;
%           end
%           m = 2*m + 1;
%           if m > e1 - e_min
%             m = e1 - e_min;
%           end
%           f1 = pow2(f1,m);
%           e1 = e1 - m;
%           if f1 == floor(f1)
%             break % f1 is integer
%           else
%             f0 = f1; e0 = e1;
%           end
%         end
%         
%         while e0 - e1 > 1
%           m = floor((e0 - e1)/2);
%           ff = pow2(f0,m);
%           ee = e0-m;
%           if ff > floor(ff)
%             f0 = ff; e0 = ee;
%           else
%             f1 = ff; e1 = ee;
%           end
%         end
%         f(k) = floor(f1);
%         e(k) = e1;
%       end
%       if ~isequal(f,f_ref) || ~isequal(e,e_ref)
%         error('ref does not match');
%       end
    end
    
    function [f,e] = writeNumberGPU(val)
      e_min = gpuArray(-1022);
      
      % gpuArray does not support [f,e] = log2(val);
      %         [f_ref, e_ref] = log2(gather(val));
      e = nextpow2(val);
      f = val ./ pow2(e);
      eq_ind = find(f==1);
      e(eq_ind) = e(eq_ind)+1;
      f(eq_ind) = 0.5;
      %         if norm(f_ref .* pow2(e_ref) - f.*pow2(e))
      %           error('GPU encode failed');
      %         end
      
      for k=1:numel(val)
        f0 = f(k);  e0 = e(k);
        
        % [f0,e0] is a mantissa-exponent pair representing val(k).
        % We try to find an equivalent pair [f1,e1] such that
        % f1 is integer. If we cannot find such a pair we let [f1,e1]
        % be such that f1 e1 = e_min.
        f1 = f0; e1 = e0;
        m = gpuArray(0);
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
    end
    
    
  end
    
end

