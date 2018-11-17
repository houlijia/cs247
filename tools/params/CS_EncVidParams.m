classdef CS_EncVidParams < CS_EncParams
  %CS_EncParams Specifies the parameters of CS encoder for ficeo
  
  properties
    start_frame=1; %Frame number to start reading the video at
    
    %the number of video frames read in (-1 = all)
    n_frames=-1;
    
    %the size of the video block tested
    blk_size = zeros(1,3);
    
    %ovrlap between adjacent video blocks (in Y component)
    blk_ovrlp = zeros(1,3)
    
    % If non-zero it overrides blk_size(3) and specifies the block
    % duration in seconds (double). The actual number of frames is
    % rounded up.
    blk_dur = 0;
    
    % If >0 it overrides blk_size(1:2) and blk_ovrlp(1:2). These are
    % set according to SAV conventions. The value indicates the number
    % of levels. If 0, the U,V color components are interpolated into the
    % same size as the Y component when the frame is read. Otherwise,
    % they are left in different sizes.
    sav_levels = 0;
    
    % type of window:
    %   1=Hanning,
    %   2=Triangular,
    %   3=sqrt(Hanning)
    %   4=sqrt(Triangular)
    wnd_type = CS_EncParams.WND_TRIANGLE;
    
    % Block backward zero extension (in Y component)
    zero_ext_b = zeros(1,3);
    
    % Block forward zero extension (in Y component)
    zero_ext_f = zeros(1,3);
    
    % Block extension by interpolated wrap around (i.e. interpolation
    % between first and last pixels in each dimension)
    % Note: In each dimension if wrap_ext is non-zero, zero_ext_b and
    % zero_ext_f must be zero.
    wrap_ext = zeros(1,3);
    
    % Specify along which dimensions each block is differenced and how many
    % times
    blk_pre_diff = [0, 0, 0];
    
    %required shifts for convolution matrices
    conv_rng = [1 1 1];
    
    % If true, mean andstandard deviation is computed by sorting the measurements,
    % removing the fraction at the ends that should be discarded based on
    % Guassian distribution assumptions, computing the variance based on the
    % remaining measurement and correcting for Gaussian distribution only.
    % Otherwise mean and standard deviation are computed for all measurements.
    qntzr_only_in_rng=false;
    
    % If non zero, blocks are processed in groups of par_blks * whole frames,
    % that is par_blks * number of blocks in a frame.
    par_blks = 0;
  end
  
  properties  % Uncoded properties
    % Decoder parameters
    dec_opts = [];
  end
  
  methods
    function obj=CS_EncVidParams(def)
      if nargin > 0
        args = {def};
      else
        args = {};
      end
      obj = obj@CS_EncParams(args{:});
    end
    
    function eql = isEqual(obj, other)
      if ~strcmp(class(obj), class(other))
        eql = false;
        return;
      elseif all(eq(obj,other))
        eql = true;
        return
      end
      otr = other.copy();
      if obj.blk_dur > 0
        otr.blk_size(3) = obj.blk_size(3);
        otr.blk_ovrlp(3) = obj.blk_ovrlp(3);
      end
      if obj.sav_levels > 0
        otr.blk_size(1:2) = obj.blk_size(1:2);
        otr.blk_ovrlp(1:2) = obj.blk_ovrlp(1:2);
      end
      
      eql = obj.isEqual@CodeElement(otr);
    end
    
    function sd = randomSeed(obj, blk_indx, blk_cnt)
      % Compute the random seed of a block with index blk_indx, if there
      % are blk_cnt blocks in the video (blk_indx and blk_cnt are array
      % of the form [V,H,T])
      
      blk_indx = blk_indx - [1,1,1];
      if obj.random.rpt_temporal
        blk_indx(3) = mod(blk_indx(3), obj.random.rpt_temporal);
      end
      if obj.random.rpt_spatial
        sd = obj.random.seed + blk_indx(3);
      else
        sd = obj.random.seed + blk_indx(1) + ...
          blk_cnt(1)*(blk_indx(2) +blk_cnt(2)*blk_indx(3));
      end
      sd = struct('seed',sd,'type',obj.random.type);
    end
    
    function str=idStr(obj, fmt)
      % Generate an ID string. The ID string is the input fmt string with
      % substitutions as follows:
      %   <Fs> - Start frame
      %   <Fn> - number of frames ('inf' for infinite).
      %   <Bp> - par_blks;
      %   <Bs> - blk_size as v-h-t. If blk_dur is non zero, t is followd
      %          by 's'. If sav_levels>0, h-t is replaced by 'L' and
      %          number of levels
      %   <Bo> - blk_ovrlp as v-h-6
      %   <Bd> - blk_pre_diff as v-h-6
      %   <Wt> - wnd_type
      %   <Zb> - zero_ext_b, as v-h-t
      %   <Zf> - zero_ext_f, as v-h-t
      %   <Zw> - wrap_ext as v-h-t
      %   <C> - C if process_color is true, B otherwise.
      %   <Mt> - Matrix type (string), omitting 'SensingMatrix' prefix.
      %   <Ma> - Matrix type and argument, as a JSON string (with '"' dropped
      %          and : replaced by ~).
      %   <Mc> - conv_mode (number)
      %   <Mr> - measurement input ratio. If there is a number
      %          before the '>', e.g.<Mr3> it is precision. Otherwise
      %          the precision is 2 digit after the decimal point.
      %   <cr> - conv_rng as v-h-t
      %   <Qm> - qntzr_wdth_mltplr.  If there is a number
      %          before the '>', e.g.<Qm3> this is precision. Otherwise
      %          the precision is 1 digit after the decimal point.
      %   <Qe> - qntzr_wdth_rng_exp.
      %   <Qa> - qntzr_ampl_stddev.  If there is a number
      %          before the '>', e.g.<Qa3> this is precision. Otherwise
      %          the precision is 1 digit after the decimal point.
      %   <Qr> - q_outrange_action_name. D for Q_DISCARD, S for Q_SAVE.
      %   <Qi> - qntzr_outrange_action. F or T for false or true
      %   <Qo> - qntzr_only_in_rng. F or T for false or true
      %   <Lc> - Lossless coder: I for LLC_INT, A for LLC_AC.
      %   <Lg> - lossless_coder_AC_gaus_thrsh.If there is a number after the
      %          before the '>', e.g.<Lg3> this is precision. Otherwise
      %          the precision is 1 digit after the decimal point.
      %   <Lm> - lossless_coder_AC_hist_type value
      %   <Nn> - case_no
      %   <Nc> - n_cases
      %   <Rd> - random.seed
      %   <Rs> - random.rpt_spatial
      %   <Rt> - Random.rpt_temporal
      %   <Sp> - Single precision as true or false
      %   *    - Date and time as yyyymmdd-HHMM
      str = regexprep(fmt, '*', datestr(now,'yyyymmdd-HHMM'));
      
      str = regexprep(str,'[<]Fs[>]', int2str(obj.start_frame));
      str = regexprep(str,'[<]Fn[>]', int2str(obj.n_frames));
      if regexp(str,'[<]Bs[>]')
        if obj.sav_levels > 0
          vh_str = int2str(obj.blk_size(1:2));
        else
          vh_str = sprintf('L%d', obj.sav_levels);
        end
        if obj.blk_dur == 0
          t_str = int2str(obj.blk_size(3));
        else
          t_str = sprintf('%fs', obj.blk_dur);
        end
        vht_str = regexprep([vh_str ' ' t_str], '\s+','-');
        str = regexprep(str,'[<]Bs[>]', vht_str);
      end
      str = regexprep(str,'[<>Bp[>]', int2str(obj.par_blks));
      str = regexprep(str,'[<]Bs[>]', int_seq(obj.blk_size));
      str = regexprep(str,'[<]Bo[>]', int_seq(obj.blk_ovrlp));
      str = regexprep(str,'[<]Bd[>]', int_seq(obj.blk_pre_diff));
      str = regexprep(str,'[<]Wt[>]', int2str(obj.wnd_type));
      str = regexprep(str,'[<]Zb[>]', int_seq(obj.zero_ext_b));
      str = regexprep(str,'[<]Zf[>]', int_seq(obj.zero_ext_f));
      str = regexprep(str,'[<]Zw[>]', int_seq(obj.wrap_ext));
      str = regexprep(str,'[<]C[>]', tf_symbol(obj.process_color, 'BC'));
      str = regexprep(str,'[<]Mt[>]', regexprep(obj.msrmnt_mtrx.type,...
        '^SensingMatrix',''));
      if regexp(str,'[<]Ma[>]')
        Ma_str = mat2json(struct(...
          regexprep(obj.msrmnt_mtrx.type,'^SensingMatrix',''),...
          {obj.msrmnt_mtrx.args}));
        Ma_str = regexprep(Ma_str, '[\s"\n]','');
        Ma_str = regexprep(Ma_str, '[:]','~');
        Ma_str = regexprep(Ma_str, '^[{]','');
        Ma_str = regexprep(Ma_str, '[}]$','');
        str = regexprep(str,'[<]Ma[>]', Ma_str);
      end
      str = regexprep(str,'[<]Mc[>]', int2str(obj.conv_mode));
      str = subst_prec(str,'[<]Mr[>]', obj.msrmnt_input_ratio,2);
      str = regexprep(str,'[<]cr[>]', int_seq(obj.conv_rng));
      str = subst_prec(str,'[<]Qm[>]', obj.qntzr_wdth_mltplr,1);
      str = subst_prec(str,'[<]Qe[>]', obj.qntzr_wdth_rng_exp,1);
      str = subst_prec(str,'[<]Qa[>]', obj.qntzr_ampl_stddev,1);
      str = regexprep(str, '[<]Qr[>]', tf_symbol(...
        obj.qntzr_outrange_action, 'DS', [obj.Q_DISCARD, obj.Q_SAVE]));
      str = regexprep(str, '[<]Qi[>]', tf_symbol(...
        obj.qntzr_outrange_action, 'FT', [obj.Q_DISCARD, obj.Q_SAVE]));
      str = regexprep(str, '[<]Qo[>]', tf_symbol(...
        obj.qntzr_only_in_rng, 'FT', [false, true]));
      str = regexprep(str, '[<]Lc[>]', tf_symbol(...
        obj.lossless_coder, 'IA', [obj.LLC_INT, obj.LLC_AC]));
      str = subst_prec(str,'[<]Lg[>]', obj.lossless_coder_AC_gaus_thrsh,1);
      str = regexprep(str, '[<]Lm[>]', int2str(obj.lossless_coder_AC_hist_type));      
      str = regexprep(str, '[<]Nn[>]', int_seq(obj.case_no));
      str = regexprep(str, '[<]Nc[>]', int_seq(obj.n_cases));
      str = regexprep(str, '[<]Rd[>]', int2str(obj.random.seed));
      str = regexprep(str, '[<]Rs[>]', int2str(obj.random.rpt_spatial));
      str = regexprep(str, '[<]Rt[>]', int2str(obj.random.rpt_temporal));
      str = regexprep(str, '[<]Sp[>]', tf_symbol(...
        obj.use_single, 'FT', [false, true]));
      
      function s = int_seq(x)
        s = regexprep(int2str(x), '\s+','-');
      end
      
      function val = tf_symbol(x, tf_val, x_val)
        if nargin < 3
          x_val = [0 1];
          if nargin < 2
            tf_val = 'TF';
          end
        end
        for k=1:length(x_val)
          if x == x_val(k)
            val = tf_val(k);
            break;
          end
        end
      end
      
      function str = subst_prec(str, fmt, val, dflt)
        fmt = strrep(fmt, '[>]', '(\d*)[>]');
        [mtch, tkn] = regexp(str, fmt, 'match', 'tokens');
        
        for k=1:length(mtch)
          if isempty(tkn{k}{1}); t=int2str(dflt);
          else t=tkn{k}{1};
          end
          str = strrep(str,mtch{k},num2str(val,['%.0' t 'f']));
        end
      end
    end
    
    function len = encode(obj, code_dst, ~)
      flg_blk_size = any(obj.blk_size);
      flg_blk_ovrlp = any(obj.blk_ovrlp);
      flg_zero_wrap_ext = any(obj.zero_ext_b) || any(obj.zero_ext_f) || ...
        any(obj.wrap_ext);
      flg_pre_diff = any(obj.blk_pre_diff);
      flg_q_save = (obj.qntzr_outrange_action == CS_EncParams.Q_SAVE);
      flg_q_in_rng = obj.qntzr_only_in_rng;
      flg_llc_ac = (obj.lossless_coder == CS_EncParams.LLC_AC);
      flg_single = obj.use_single;
      flg_rnd_sptl = obj.random.rpt_spatial;
      flg_blk_dur = (obj.blk_dur > 0);
      flg_mtrx_args = ~isempty(fieldnames(obj.msrmnt_mtrx.args));
      flg_mtrx_nxt = isfield(obj.msrmnt_mtrx, 'nxt') ...
        && ~isempty(obj.msrmnt_mtrx.nxt);
      flg_qntzr_exp_rng = (obj.qntzr_wdth_rng_exp ~= ...
        CS_EncParams.QNTZR_WDTH_RNG_EXP_DFLT);
      flags = [obj.use_json, obj.process_color, flg_blk_size, flg_blk_ovrlp,...
        flg_zero_wrap_ext, flg_pre_diff, flg_q_save, flg_q_in_rng, ...
        flg_llc_ac, flg_single, flg_rnd_sptl, flg_blk_dur, ...
        flg_mtrx_args, flg_mtrx_nxt, flg_qntzr_exp_rng];
      len = code_dst.writeBitsArray(flags);
      if ischar(len); return; end
      
      len0 = obj.encodeCommon(code_dst, flg_mtrx_args, flg_mtrx_nxt, ...
        flg_qntzr_exp_rng);
      if ischar(len0);
        len = len0; return
      else
        len = len+len0;
      end
      if obj.use_json
        return
      end
            
      len0 = code_dst.writeSInt([obj.n_frames, obj.sav_levels]);
      if ischar(len0);
        len = len0;
        return;
      end
      len = len + len0;
      
      int_vals = [obj.start_frame...
        obj.wnd_type ...
        size(obj.conv_rng,1) ...
        obj.random.rpt_temporal ...
        obj.par_blks];
      len0 = code_dst.writeUInt(int_vals);
      if ischar(len0); len = len0; return; end;
      len = len + len0;
      
      len0 = code_dst.writeUInt(obj.conv_rng);
      if ischar(len0); len = len0; return; end;
      len = len + len0;
      
      int3_vals = zeros(0,3);
      if(flg_blk_size)
        int3_vals = [ int3_vals; obj.blk_size ];
      end
      if(flg_blk_ovrlp)
        int3_vals = [ int3_vals; obj.blk_ovrlp ];
      end
      if(flg_zero_wrap_ext)
        int3_vals = [ int3_vals; obj.zero_ext_b; obj.zero_ext_f; obj.wrap_ext ];
      end
      if(flg_pre_diff)
        int3_vals = [ int3_vals; obj.blk_pre_diff ];
      end
      if ~isempty(int3_vals)
        len0 = code_dst.writeUInt(int3_vals);
        if ischar(len0); len = len0; return; end;
        len = len + len0;
      end
      
      len0 = code_dst.writeUInt([obj.case_no obj.n_cases]);
      if ischar(len0); len = len0; return; end;
      len = len + len0;
      
      if flg_blk_dur
        len0 = code_dst.writeNumber(obj.blk_dur);
        if ischar(len0); len = len0; return; end;
        len = len + len0;
      end
    end
    
    function len = decode(obj, code_src, ~, cnt)
      if nargin < 4; cnt = inf; end
      
      [flags, n_rd] = code_src.readBitsArray(15, cnt);
      if ischar(flags) || (isscalar(flags) && flags == -1)
        len = flags;
        return;
      else
        len = n_rd;
        cnt = cnt - n_rd;
        flags = logical(flags);
        flg_use_json = flags(1);
        flg_process_color = flags(2);
        flg_blk_size = flags(3);
        flg_blk_ovrlp = flags(4);
        flg_zero_wrap_ext = flags(5);
        flg_pre_diff = flags(6);
        flg_q_save = flags(7);
        flg_q_in_rng = flags(8);
        flg_q_llc_ac = flags(9);
        flg_single = flags(10);
        flg_rnd_sptl = flags(11);
        flg_blk_dur = flags(12);
        flg_mtrx_args = flags(13);
        flg_mtrx_nxt = flags(14);
        flg_qntzr_exp_rng = flags(15);
      end
      
      n_rd = obj.decodeCommon(code_src, cnt, ...
        flg_use_json, flg_mtrx_args, flg_mtrx_nxt, flg_qntzr_exp_rng);
      if ischar(n_rd)
        len = n_rd; 
        return
      else
        len = len + n_rd;
        cnt = cnt - n_rd;
        if flg_use_json
          obj.blk_size = horzcat(obj.blk_size{:});
          obj.blk_ovrlp = horzcat(obj.blk_ovrlp{:});
          obj.zero_ext_b = horzcat(obj.zero_ext_b{:});
          obj.zero_ext_f = horzcat(obj.zero_ext_f{:});
          obj.wrap_ext = horzcat(obj.wrap_ext{:});
          obj.conv_rng = horzcat(obj.conv_rng{:});
          obj.blk_pre_diff = horzcat(obj.blk_pre_diff{:});
           return
        end
      end
      
      [sint_vals, n_rd] = code_src.readSInt(cnt, [1,2]);
      if ischar(sint_vals)
        if strcmp(sint_vals, '')
          sint_vals = -1;
        end
        len = sint_vals;
        return;
      else
        len = len+ n_rd;
        cnt = cnt - n_rd;
      end
      sint_vals = double(sint_vals);
      
      [int_vals, n_rd] = code_src.readUInt(cnt, [1,5]);
      if ischar(int_vals) || (isscalar(int_vals) && int_vals == -1)
        len = int_vals;
        return;
      else
        len = len+n_rd;
        cnt = cnt - n_rd;
      end
      int_vals = double(int_vals);
      l_cnv_rng = int_vals(3);

      [cnv_rng, n_rd] = code_src.readUInt(cnt, [l_cnv_rng,3]);
      if ischar(cnv_rng) || (isscalar(cnv_rng) && cnv_rng == -1)
        if ischar(cnv_rng)
          len = cnv_rng;
        else
          len = 'Unexpected end of data';
        end
        return
      else
        len = len + n_rd;
        cnt = cnt - n_rd;
      end
      cnv_rng = double(cnv_rng);
      
      l_int3_vals = flg_blk_size + flg_blk_ovrlp + 3*flg_zero_wrap_ext + flg_pre_diff;
      if l_int3_vals>0
        [int3_vals, n_rd] = code_src.readUInt(cnt, [l_int3_vals,3]);
        if ischar(int3_vals) || (isscalar(int3_vals) && int3_vals == -1)
          if ischar(int3_vals)
            len = int3_vals;
          else
            len = 'Unexpected end of data';
          end
          return
        else
          len = len + n_rd;
          cnt = cnt - n_rd;
        end
        int3_vals = double(int3_vals);
      end
      
      [case_vals, n_rd] = code_src.readUInt(cnt, [1,2]);
      if ischar(case_vals) || (isscalar(case_vals) && case_vals == -1)
        if ischar(case_vals)
          len = case_vals;
        else
          len = 'Unexpected end of data';
        end
        return
      else
        len = len + n_rd;
        cnt = cnt - n_rd;
      end
      case_vals = double(case_vals);
      
      if flg_blk_dur
        [flt_vals, n_rd] = code_src.readNumber(cnt, 1);
        if ischar(flt_vals) || (isscalar(flt_vals) && flt_vals == -1)
          if ischar(flt_vals)
            len = flt_vals;
          else
            len = 'Unexpected end of data';
          end
          return
        else
          len = len + n_rd;
        end
      end
      
      indx3 = 0;
      
      obj.n_frames = double(sint_vals(1));
      obj.start_frame = int_vals(1);
      obj.random.rpt_spatial = double(flg_rnd_sptl);
      obj.random.rpt_temporal = int_vals(4);
      obj.sav_levels = sint_vals(2);
      obj.qntzr_only_in_rng = flg_q_in_rng;
      obj.par_blks = int_vals(5);
      obj.use_single = flg_single;
      obj.process_color = flg_process_color;
      obj.wnd_type = int_vals(2);
      if flg_q_save
        obj.qntzr_outrange_action = CS_EncParams.Q_SAVE;
      else
        obj.qntzr_outrange_action = CS_EncParams.Q_DISCARD;
      end
      if flg_q_llc_ac
        obj.lossless_coder = CS_EncParams.LLC_AC;
      else
        obj.lossless_coder = CS_EncParams.LLC_INT;
      end
      obj.conv_mode = int_vals(3);
      if flg_blk_size
        indx3 = indx3 + 1;
        obj.blk_size = int3_vals(indx3, :);
      end
      if flg_blk_ovrlp
        indx3 = indx3 + 1;
        obj.blk_ovrlp = int3_vals(indx3, :);
      end
      if flg_zero_wrap_ext
        obj.zero_ext_b = int3_vals(indx3+1,:);
        obj.zero_ext_f = int3_vals(indx3+2,:);
        obj.wrap_ext = int3_vals(indx3+3,:);
        indx3 = indx3 + 3;
      end
      if flg_pre_diff
        obj.blk_pre_diff = int3_vals(indx3+1,:);
      end
      obj.case_no = case_vals(1);
      obj.n_cases = case_vals(2);
      obj.conv_rng = cnv_rng;
      if flg_blk_dur
        obj.blk_dur = flt_vals;
      end
    end
  end
  
end

