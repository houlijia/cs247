% Documentation for this function is in VidRegion.m next to the function 
% signature
function len = decode(obj, code_src, info, cnt)
    if nargin < 4
        cnt = inf;
    end
    
    [wf, len] = code_src.readUInt(cnt);
    if ischar(wf) || wf == -1
      len = wf;
      return
    end
    cnt = cnt - len;
    
    if wf    % Whole frame
      [frm_rng, len1] = code_src.readUInt(cnt, [1,2]);
      if ischar(frm_rng)
        len = frm_rng;
        return
      elseif frm_rng == -1
        len = 'EOD encountered while reading';
        return
      end
      len = len + len1;
      
      blk_cnt = info.vid_blocker.blk_cnt;
      frm_ofst = frm_rng(1);
      n_frms = frm_rng(2)+1;
      n_b = blk_cnt(1)*blk_cnt(2)*n_frms;
      blk_vals = zeros(n_b,3);
      v = blk_cnt(1);
      h = blk_cnt(2);
      t = frm_ofst;
      for k = 1:n_b
        if v==blk_cnt(1)
          v = 1;
          if h==blk_cnt(2)
            h=1;
            t = t+1;
          else
            h = h+1;
          end
        else
          v = v+1;
        end
        blk_vals(k,:) = [v,h,t];
      end
    else % whole frames
      [n_b, len1] = code_src.readUInt(cnt);
      if ischar(n_b) 
        len = n_b;
        return
      elseif n_b == -1
        len = 'EOD encountered while reading';
        return
      end
      n_b = double(n_b);
      cnt = cnt - len1;
      len = len + len1;
      
      [blk_vals, len1] = code_src.readUInt(cnt, [n_b,3]);
      if ischar(blk_vals)
        len = blk_vals;
        return
      elseif isscalar(blk_vals) && blk_vals == -1
        len = 'EOD encountered while reading';
        return
      end
      len = len + len1;
    end
    
    zero_ext = [info.enc_opts.zero_ext_b; info.enc_opts.zero_ext_f];
    wrap_ext = info.enc_opts.wrap_ext;
    obj.init(double(blk_vals), info.vid_blocker, zero_ext, wrap_ext);
end

