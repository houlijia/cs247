% Documentation for this function is in VidRegion.m next to the function 
% signature
function len = decode(obj, code_src, info, cnt)
    if nargin < 4
        cnt = inf;
    end
    
    [n_b, len] = code_src.readUInt(cnt);
    if ischar(n_b) || n_b == -1
        len = n_b;
        return
    end
    n_b = double(n_b);
    cnt = cnt - len;
    
    [blk_vals, len1] = code_src.readUInt(cnt, [n_b,3]);
    if ischar(blk_vals)
        len = blk_vals;
        return
    elseif isscalar(blk_vals) && blk_vals == -1
        len = 'EOD encountered while reading';
        return
    end
    len = len + len1;
    
    zero_ext = [info.enc_opts.zero_ext_b; info.enc_opts.zero_ext_f];
    wrap_ext = info.enc_opts.wrap_ext;
    obj.init(double(blk_vals), info.vid_blocker, zero_ext, wrap_ext);
end

