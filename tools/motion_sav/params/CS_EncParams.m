classdef CS_EncParams  < ProcessingParams & CodeElement
    %CS_EncParams Specifies the parameters of CS encoder
    
    properties (Constant=true)
        % Codes specifiying action when a measurement is out of the range of the
        % quantizer:
        Q_DISCARD=1;  % Replace quantization code by a special "lost" code.
        Q_SAVE=2;     % Replace quantization code by a special "lost" code and
        % provide the unquantized values.
        
        q_outrange_action_name = {...
            'Q_DISCARD - discard out of range measurements',...
            'Q_SAVE - Save unquantized value'};
        
        
        % Codes for the type of lossless coder used:
        LLC_INT=1;    % Represent quantized or unquantized measurement as integers
        LLC_AC=2;     % Use arithmetic coding to repersent measurements
        
        % Lossless coder names:
        llc_names = { 'LLC_INT - code as integers', 'LLC_AC - arithmetic coding'};
        
        %     entropy_coding_strategy_names = {...
        %        'no_quantization', 'quantization_only', 'quantization_and_arithmetic_coding';...
        %        'noQ',         'QOnly',         'QandAC'};
        
        % Window types
        WND_NONE = 0;
        WND_HANNING=1;
        WND_TRIANGLE=2;
        WND_SQRT_HANNING=3
        WND_SQRT_TRIANGLE=4;
        
        wnd_types = [CS_EncParams.WND_NONE, CS_EncParams.WND_HANNING, ...
          CS_EncParams.WND_TRIANGLE,... 
          CS_EncParams.WND_SQRT_HANNING, CS_EncParams.WND_SQRT_TRIANGLE];
    end
    
    
    properties (SetAccess=protected)
        start_frame=1; %Frame number to start reading the video at
        
        %the number of video frames read in (-1 = all)
        n_frames=-1;
        
      % If true process YUV, otherwise process only Y
        process_color = true;
        
        % Specify random number generator for each block. The fields are:
        %   seed - Initial seed of generator
        %   rpt_spatial - if 0, the seed is changed for each block
        %                 spatially (increased by 1.  If 1, the seed is
        %                 kept the same for all blocks of the same frames
        %                 group.
        %   rpt_temporal - If 0 the seed changes for each block temporally
        %                 (increased by B when the block in the same
        %                 spatial postion appears again, where B=1 if
        %                 rpt_spatial=0 and B=number of blocks per frame
        %                 if rpt_spatial=1). Otherwise, rpt_temporal is
        %                 the periodicity in which the same seed is
        %                 repeated. In particular, rpt_temporal=1 means
        %                 that all blocks in the same spatial position have
        %                 the same random seed.
        random=struct(...
            'seed',0,...
            'type',0,...
            'rpt_spatial',0,...
            'rpt_temporal',0 ...
        );
        
        %the size of the video block tested
        blk_size = zeros(1,3);
        
        %ovrlap between adjacent video blocks (in Y component)
        blk_ovrlp = zeros(1,3)
                
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
        
        % measurement matrix definition.  A struct with fields:
        %   type - (character string type of matrix
        %   args - (struct) - arguments used to define the matrix
        msrmnt_mtrx = struct('type','SensingMatrixWH', 'args', struct());
        
        % Mode of convolutional matrix
        conv_mode=SensingMatrixConvolve.MODE_GLBL;
        
        %required shifts for convolution matrices
        conv_rng = [1 1 1];
        
        %input parameters to the compressed sensed video codec
        msrmnt_input_ratio=1.0;
        
        % The step size of the quantizer, normalized by the standard deviation of
        % the the pixel quantization noise in measurements.  0 indicates no quantization
        qntzr_wdth_mltplr=1.0;
        
        % The amplitude of the quantizer, normalized by the standard deviation of
        % the measurements.  If no quantization is done (qntzr_wdth_mltplr = 0) the
        % normalization is by 1 (no normalization).
        qntzr_ampl_stddev=3.0;
        
        % Code for action for out of range values. is Q_DISCARD, Q_SAVE.
        qntzr_outrange_action=CS_EncParams.Q_DISCARD;
        
        % Specify along which dimensions each block is differenced and how many
        % times
        blk_pre_diff = [0, 0, 0];
        
        % Losslss coder selection
        lossless_coder=CS_EncParams.LLC_INT;
        
        % If lossless coder is LLC_AC, this parameter determines when the label
        % frequency histogram is approximated by Gaussian distribution.  This
        % happens when the ratio between the quantizer step and the quantizer
        % amplitude is less than the threshold lossless_coder_AC_gaus_thrsh,
        % or equivalently when
        % quantizer normalized amplitude / obj.n_bins < obj.lossless_coder_AC_gaus_thrsh
        lossless_coder_AC_gaus_thrsh = 0.1;
        
    end
    
    properties  % Uncoded properties
        % Decoder parameters
        dec_opts = [];
    end
    
    methods
        function obj=CS_EncParams(def)
            if nargin > 0
                args = {def};
            else
                args = {};
            end
            obj = obj@ProcessingParams(args{:});
            
            if ~any(obj.wnd_types == obj.wnd_type)
              error('Illegal wnd_type: %d', obj.wnd_type);
            end
            
            if any(all([any([obj.zero_ext_f;obj.zero_ext_b]);obj.wrap_ext]),2)
              error(['if wrap_ext is non-zero in any dimension, zero_ext_f'...
                'and zero_ext_b cannot be non-zero in the same dimension']);
            end
        end
        
        % Compute the random seed of a block with index blk_indx, if there
        % are blk_cnt blocks in the video (blk_indx and blk_cnt are array
        % of the form [V,H,T])
        
        function sd = randomSeed(obj, blk_indx, blk_cnt)
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
        
        function str = classDescription(~)
          str = 'Encoding options';
        end
        
        
        % setParams sets the parameters of the object. It also clears code
        % if a parameter was actually changed.
        % Input
        %   obj - this object
        %   params - can be:
        %         a struct whose field names are the same as some of the
        %            properties of obj.  The struct values are going to be set
        %            accordingly. If a field value is a string and it begins with
        %            an ampersand (&), the ampersand is removed and the field value
        %            is evalueated before being assigned.
        %         An object of a type which is a superclass of obj.  The
        %            non-constant fields in params are copied to obj.
        function setParams(obj, params)
            changed = false;
            if isstruct(params)
                flds = fieldnames(params);
                for k=1:length(flds)
                    fld = flds{k};
                    if ~changed && ~isequal(obj.(fld), params.(fld))
                        changed = true;
                    end
                    obj.(fld) = setval(obj.(fld),params.(fld));
                end
            else
                mp = metaclass(params);
                obj_props = properties(obj);
                for k=1:length(mp.PropertyList)
                    prop = mp.PropertyList(k);
                    if ~any(strcmp(obj_props, prop.Name))
                        continue;
                    end
                    if ~prop.Constant
                        if ~changed &&...
                                ~isequal(obj.(prop.Name), params.(prop.Name))
                            changed = true;
                        end
                        obj.(prop.Name) = params.(prop.Name);
                    end
                end
            end
            
            if changed
                obj.code = 0;
            end
            
            function d = setval(d,v)
                if isstruct(d)
                    fls = fields(v);
                    for j=1:length(fls)
                        if isfield(d, fls{j})
                            d.(fls{j}) = setval(d.(fls{j}),v.(fls{j}));
                        else
                            d.(fls{j}) = v.(fls{j});
                        end
                    end
                else
                    d = v;
                end
            end
            
        end
        
        % Generate an ID string. The ID string is the input fmt string with
        % substitutions as follows:
        %   <Fs> - Start frame
        %   <Fn> - number of frames ('inf' for infinite).
        %   <Bs> - blk_size as v-h-t
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
        %   <Mr> - measurement input ratio. If there is a number after the 
        %          before the '>', e.g.<Mr3> it is precision. Otherwise
        %          the precision is 2 digit after the decimal point.
        %   <cr> - conv_rng as v-h-t
        %   <Qm> - qntzr_wdth_mltplr.  If there is a number after the 
        %          before the '>', e.g.<Qm3> this is precision. Otherwise
        %          the precision is 1 digit after the decimal point.
        %   <Qa> - qntzr_ampl_stddev.  If there is a number after the 
        %          before the '>', e.g.<Qa3> this is precision. Otherwise
        %          the precision is 1 digit after the decimal point.
        %   <Qr> - q_outrange_action_name. D for Q_DISCARD, S for Q_SAVE.
        %   <Lc> - Lossless coder: I for LLC_INT, A for LLC_AC.
        %   <Lg> - lossless_coder_AC_gaus_thrsh.If there is a number after the 
        %          before the '>', e.g.<Lg3> this is precision. Otherwise
        %          the precision is 1 digit after the decimal point.
        %   <Nn> - case_no
        %   <Nc> - n_cases
        %   <Rd> - random.seed
        %   <Rs> - random.rpt_spatial
        %   <Rt> - Random.rpt_temporal
        %   *    - Date and time as yyyymmdd-HHMM
        function str=idStr(obj, fmt)
            str = regexprep(fmt, '*', datestr(now,'yyyymmdd-HHMM'));
            
            str = regexprep(str,'[<]Fs[>]', int2str(obj.start_frame));
            str = regexprep(str,'[<]Fn[>]', int2str(obj.n_frames));
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
            str = subst_prec(str,'[<]Qa[>]', obj.qntzr_ampl_stddev,1); 
            str = regexprep(str, '[<]Qr[>]', tf_symbol(...
                obj.qntzr_outrange_action, 'DS', [obj.Q_DISCARD, obj.Q_SAVE]));
            str = regexprep(str, '[<]Lc[>]', tf_symbol(...
                obj.lossless_coder, 'IA', [obj.LLC_INT, obj.LLC_AC]));
            str = subst_prec(str,'[<]Lg[>]', obj.lossless_coder_AC_gaus_thrsh,1);
            str = regexprep(str, '[<]Nn[>]', int_seq(obj.case_no));
            str = regexprep(str, '[<]Nc[>]', int_seq(obj.n_cases));
            str = regexprep(str, '[<]Rd[>]', int2str(obj.random.seed));
            str = regexprep(str, '[<]Rs[>]', int2str(obj.random.rpt_spatial));
            str = regexprep(str, '[<]Rt[>]', int2str(obj.random.rpt_temporal));
            
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
            len = code_dst.writeSInt(obj.n_frames);
            if ischar(len); return; end
            
            len0 = code_dst.writeUInt(...
                [obj.start_frame obj.random.seed double(obj.process_color)...
                obj.wnd_type ...
                obj.qntzr_outrange_action obj.lossless_coder...
                obj.conv_mode length(obj.conv_rng)...
                obj.random.rpt_spatial, obj.random.rpt_temporal]);
            if ischar(len0); len = len0; return; end;
            len = len + len0;
            
            len0 = code_dst.writeUInt(obj.conv_rng);
            if ischar(len0); len = len0; return; end;
            len = len + len0;
            
            len0 = code_dst.writeUInt([obj.blk_size; obj.blk_ovrlp;...
                obj.zero_ext_b; obj.zero_ext_f; obj.wrap_ext; ...
                obj.blk_pre_diff]);
            if ischar(len0); len = len0; return; end;
            len = len + len0;
            
            len0 = code_dst.writeUInt([obj.case_no obj.n_cases]);
            if ischar(len0); len = len0; return; end;
            len = len + len0;
            
            len0 = code_dst.writeNumber([obj.msrmnt_input_ratio, ...
                obj.qntzr_wdth_mltplr, obj.qntzr_ampl_stddev, ...
                obj.lossless_coder_AC_gaus_thrsh]);
            if ischar(len0); len = len0; return; end;
            len = len + len0;
            
            json_mtrx = mat2json(obj.msrmnt_mtrx);
            len0 = code_dst.writeString(json_mtrx);
            if ischar(len0); len = len0; return; end;
            len = len + len0;
        end
        
        function len = decode(obj, code_src, ~, cnt)
            if nargin < 4; cnt = inf; end
            
            [n_frm, n_rd] = code_src.readSInt(cnt);
            if ischar(n_frm) || (isscalar(n_frm) && n_frm == -1)
                len = n_frm;
                return;
            else
                len = n_rd;
                cnt = cnt - n_rd;
            end
            
            [int_vals, n_rd] = code_src.readUInt(cnt, [1,10]);
            if ischar(int_vals) || (isscalar(int_vals) && int_vals == -1)
                len = int_vals;
                return;
            else
                len = len+n_rd;
                cnt = cnt - n_rd;
            end
            int_vals = double(int_vals);
            l_cnv_rng = int_vals(8);
            
            [cnv_rng, n_rd] = code_src.readUInt(cnt, [1,l_cnv_rng]);
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
            
            [int3_vals, n_rd] = code_src.readUInt(cnt, [6,3]);
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
            
            [flt_vals, n_rd] = code_src.readNumber(cnt, [1,4]);
            if ischar(flt_vals) || (isscalar(flt_vals) && flt_vals == -1)
                if ischar(flt_vals)
                    len = flt_vals;
                else
                    len = 'Unexpected end of data';
                end
                return
            else
                len = len + n_rd;
                cnt = cnt - n_rd;
            end
            
            [mtrx_spec, n_rd, err_msg] = code_src.readString(cnt);
            if ~ischar(mtrx_spec)
                if mtrx_spec == 0
                    len = err_msg;
                else
                    len = 'Unexpected end of data';
                end
                return
            else
                len = len + n_rd;
            end
            
            obj.n_frames = n_frm;
            obj.start_frame = int_vals(1);
            obj.random.seed = int_vals(2);
            obj.random.rpt_spatial = int_vals(9);
            obj.random.rpt_temporal = int_vals(10);
            obj.process_color = (int_vals(3) ~= 0);
            obj.wnd_type = int_vals(4);
            obj.qntzr_outrange_action = int_vals(5);
            obj.lossless_coder =  int_vals(6);
            obj.conv_mode = int_vals(7);
            obj.blk_size = int3_vals(1,:);
            obj.blk_ovrlp = int3_vals(2,:);
            obj.zero_ext_b = int3_vals(3,:);
            obj.zero_ext_f = int3_vals(4,:);
            obj.wrap_ext = int3_vals(5,:);
            obj.blk_pre_diff = int3_vals(6,:);
            obj.case_no = case_vals(1);
            obj.n_cases = case_vals(2);
            obj.conv_rng = cnv_rng;
            obj.msrmnt_input_ratio = flt_vals(1);
            obj.qntzr_wdth_mltplr = flt_vals(2);
            obj.qntzr_ampl_stddev = flt_vals(3);
            obj.lossless_coder_AC_gaus_thrsh = flt_vals(4);
            obj.msrmnt_mtrx = parse_json(mtrx_spec);
        end
        
    end
    
    
end

