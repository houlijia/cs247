classdef CS_EncImgParams < CS_EncParams
  %CS_EncParams Specifies the parameters of CS encoder for ficeo
  
  properties (Constant=true)
    start_frame=1; %Frame number to start reading the video at
    
    %the number of video frames read in (-1 = all)
    n_frames=1;
    
    sav_levels = 0;
    blk_dur = 0;
    blk_ovrlp = [0,0,0];
    zero_ext_b = zeros(1,3);
    zero_ext_f = zeros(1,3);
    wrap_ext = zeros(1,3);
    blk_pre_diff = [0, 0, 0];
    wnd_type = CS_EncParams.WND_TRIANGLE;
    qntzr_only_in_rng=false;
    par_blks = 0;
  end
  
  properties (SetAccess=protected)
    blk_size = [0,0,1]; % Serves here is image size
    bit_depth = 8; % bits per pixel
  end
  
  methods
    function obj=CS_EncImgParams(def)
      if nargin > 0
        args = {def};
      else
        args = {};
      end
      obj = obj@CS_EncParams(args{:});
    end
    
    function setImgSize(obj, sz, bit_depth_)
      % Set the image size. sz is the size of the pixel array, [v,h,c] where v
      % is the vertical size (number of rows), h is the horizontal size (number
      % of columns) and c is the number of columns (1 or 3).
      sz = [sz, ones(1,3-length(sz))];
      obj.blk_size = [sz(1:2) 1];
      obj.process_color = obj.process_color && (sz(3) > 1);
      if nargin >= 3
        obj.bit_depth = min(obj.bit_depth, bit_depth_);
      end
    end
    
    function str=idStr(obj, fmt)
      % Generate an ID string. The ID string is the input fmt string with
      % substitutions as follows:
      %   <Bp> - par_blks;
      %   <C> - C if process_color is true, B otherwise.
      %   <Mt> - Matrix type (string), omitting 'SensingMatrix' prefix.
      %   <Ma> - Matrix type and argument, as a JSON string (with '"' dropped
      %          and : replaced by ~).
      %   <Mc> - conv_mode (number)
      %   <Mr> - measurement input ratio. If there is a number
      %          before the '>', e.g.<Mr3> it is precision. Otherwise
      %          the precision is 2 digit after the decimal point.
      %   <Qm> - qntzr_wdth_mltplr.  If there is a number
      %          before the '>', e.g.<Qm3> this is precision. Otherwise
      %          the precision is 1 digit after the decimal point.
      %   <Qe> - qntzr_wdth_rng_exp.
      %   <Qa> - qntzr_ampl_stddev.  If there is a number
      %          before the '>', e.g.<Qa3> this is precision. Otherwise
      %          the precision is 1 digit after the decimal point.
      %   <Qr> - q_outrange_action_name. D for Q_DISCARD, S for Q_SAVE.
      %   <Qi> - qntzr_outrange_action. F or T for false or true
      %   <Lc> - Lossless coder: I for LLC_INT, A for LLC_AC.
      %   <Lg> - lossless_coder_AC_gaus_thrsh.If there is a number after the
      %          before the '>', e.g.<Lg3> this is precision. Otherwise
      %          the precision is 1 digit after the decimal point.
      %   <Lm> - lossless_coder_AC_hist_type value
      %   <Rd> - random.seed
      %   <Rs> - random.rpt_spatial
      %   <Sp> - Single precision as true or false
      %   *    - Date and time as yyyymmdd-HHMM
      str = regexprep(fmt, '*', datestr(now,'yyyymmdd-HHMM'));
      
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
      str = subst_prec(str,'[<]Qm[>]', obj.qntzr_wdth_mltplr,1);
      str = subst_prec(str,'[<]Qe[>]', obj.qntzr_wdth_rng_exp,1);
      str = subst_prec(str,'[<]Qa[>]', obj.qntzr_ampl_stddev,1);
      str = regexprep(str, '[<]Qr[>]', tf_symbol(...
        obj.qntzr_outrange_action, 'DS', [obj.Q_DISCARD, obj.Q_SAVE]));
      str = regexprep(str, '[<]Qi[>]', tf_symbol(...
        obj.qntzr_outrange_action, 'FT', [obj.Q_DISCARD, obj.Q_SAVE]));
      str = regexprep(str, '[<]Lc[>]', tf_symbol(...
        obj.lossless_coder, 'IA', [obj.LLC_INT, obj.LLC_AC]));
      str = subst_prec(str,'[<]Lg[>]', obj.lossless_coder_AC_gaus_thrsh,1);
      str = regexprep(str, '[<]Lm[>]', int2str(obj.lossless_coder_AC_hist_type));      
      str = regexprep(str, '[<]Rd[>]', int2str(obj.random.seed));
      str = regexprep(str, '[<]Rs[>]', int2str(obj.random.rpt_spatial));
      str = regexprep(str, '[<]Sp[>]', tf_symbol(...
        obj.use_single, 'FT', [false, true]));
      
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
      flg_q_save = (obj.qntzr_outrange_action == CS_EncParams.Q_SAVE);
      flg_llc_ac = (obj.lossless_coder == CS_EncParams.LLC_AC);
      flg_single = obj.use_single;
      flg_rnd_sptl = obj.random.rpt_spatial;
      flg_mtrx_args = ~isempty(fieldnames(obj.msrmnt_mtrx.args));
      flg_mtrx_nxt = isfield(obj.msrmnt_mtrx, 'nxt') ...
        && ~isempty(obj.msrmnt_mtrx.nxt);
      flg_qntzr_exp_rng = (obj.qntzr_wdth_rng_exp ~= ...
        CS_EncParams.QNTZR_WDTH_RNG_EXP_DFLT);
      flags = [obj.use_json, obj.process_color, flg_q_save, ...
        flg_llc_ac, flg_single, flg_rnd_sptl, flg_mtrx_args, flg_mtrx_nxt,...
        flg_qntzr_exp_rng];
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
      
      int_vals = [obj.blk_size(1:2) obj.bit_depth];
      len0 = code_dst.writeUInt(int_vals);
      if ischar(len0); len = len0; return; end;
      len = len + len0;
    end
    
    function len = decode(obj, code_src, ~, cnt)
      if nargin < 4; cnt = inf; end
      
      [flags, n_rd] = code_src.readBitsArray(9, cnt);
      if ischar(flags) || (isscalar(flags) && flags == -1)
        len = flags;
        return;
      else
        len = n_rd;
        cnt = cnt - n_rd;
        flags = logical(flags);
        flg_use_json = flags(1);
        flg_process_color = flags(2);
        flg_q_save = flags(3);
        flg_q_llc_ac = flags(4);
        flg_single = flags(5);
        flg_rnd_sptl = flags(6);
        flg_mtrx_args = flags(7);
        flg_mtrx_nxt = flags(8);
        flg_qntzr_exp_rng = flags(9);
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
          return
        end
      end
      
      [int_vals, n_rd] = code_src.readUInt(cnt, [1,3]);
      if ischar(int_vals) || (isscalar(int_vals) && int_vals == -1)
        len = int_vals;
        return;
      else
        len = len+n_rd;
      end
      int_vals = double(int_vals);
      
      obj.random.rpt_spatial = double(flg_rnd_sptl);
      obj.use_single = flg_single;
      obj.process_color = flg_process_color;
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
      obj.blk_size(1:2) = int_vals(1:2);
      obj.bit_depth = int_vals(3);
    end
  end
  
end

