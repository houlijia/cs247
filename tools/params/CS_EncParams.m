classdef CS_EncParams  < ProcessingParams & CodeElement
    %CS_EncParams Specifies the parameters of CS encoder
    
    properties (Constant=true)
        % Codes specifiying action when a measurement is out of the range of the
        % quantizer:
        Q_DISCARD=0;  % Replace quantization code by a special "lost" code.
        Q_SAVE=1;     % Replace quantization code by a special "lost" code and
        
        % Codes specifying how to interpret the qntzr_wdth_mltplr parameter
        Q_WDTH_ABS = 0; % quantizer step is the actual value of the paramter.
        Q_WDTH_NRML = 1; % quantizer step = paramter * digitization noise std.dev.
        Q_WDTH_CSR = 2; % quantizer step = paramter *
                        %    (digitization noise std.dev. / measurement_input_ratio) 
        Q_WDTH_RNG = 3; % quantizer step = paramter * EST_rng, where 
                        %   EST_RNG = 
                        %     mean(measurements .^ (qntzr_wdth_rng_exp)^(1/qntzr_wdth_rng_exp)
        Q_WDTH_RNG_CSR = 4; % quantizer step = paramter * (EST_RNG/ measurement_input_ratio) 
                        %    where 
                        %     EST_RNG = 
                        %        mean(measurements .^ (qntzr_wdth_rng_exp)^(1/qntzr_wdth_rng_exp)
                        
       QNTZR_WDTH_RNG_EXP_DFLT = 4;
        
       % Codes for the type of lossless coder used:
        LLC_INT=1;    % Represent quantized or unquantized measurement as integers
        LLC_AC=2;     % Use arithmetic coding to repersent measurements
        
        % Codes for type of histogram coding in AC lossless coder
        LLC_AC_HST_FULL=1;  % Single histogram with full representation
        LLC_AC_HST_FLEX=2;  % Single histogram with flexible representation
        LLC_AC_HST_MLTI=3;  % Multiple flexible histograms
        
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
      % If true, the object is encoded as a JSON string.
      use_json = false;
      
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
      
      % measurement matrix definition.  A struct with fields:
      %   type - (character string type of matrix
      %   args - (struct) - arguments used to define the matrix
      % If type is 'NONE' no matrix is defined (no encoding).
      msrmnt_mtrx = struct('type','SensingMatrixWH', 'args', struct());
      
      % Mode of convolutional matrix
      conv_mode=SensingMatrixSqr.SLCT_MODE_ARBT;
      
      %input parameters to the compressed sensed video codec
      msrmnt_input_ratio=1.0;
      
      % mode of qntzr_wdth_mltplr interpretation
      qntzr_wdth_mode = CS_EncParams.Q_WDTH_NRML;
      
      % The step size of the quantizer, normalized by the standard deviation of
      % the pixel quantization noise in measurements.  0 indicates no quantization
      qntzr_wdth_mltplr=1.0;
      
      % if q_wdth_mode>=Q_WDTH_RNG, this is the exponent used in the stimated
      % range
      qntzr_wdth_rng_exp = CS_EncParams.QNTZR_WDTH_RNG_EXP_DFLT;
      

      % The amplitude of the quantizer, normalized by the standard deviation of
      % the measurements.  If no quantization is done (qntzr_wdth_mltplr = 0) the
      % normalization is by 1 (no normalization).
      qntzr_ampl_stddev=3.0;
      
      % Code for action for out of range values. is Q_DISCARD, Q_SAVE.
      qntzr_outrange_action=CS_EncParams.Q_DISCARD;
      
      % Losslss coder selection
      lossless_coder=CS_EncParams.LLC_INT;
      
      % If lossless coder is LLC_AC, this parameter determines when the label
      % frequency histogram is approximated by Gaussian distribution.  This
      % happens when the ratio between the quantizer step and the quantizer
      % amplitude is less than the threshold lossless_coder_AC_gaus_thrsh,
      % or equivalently when
      % quantizer normalized amplitude / obj.n_bins < obj.lossless_coder_AC_gaus_thrsh
      % A value of -1 specifies to check and choose the best between the two
      % options.
      lossless_coder_AC_gaus_thrsh = -1; 
      
      % Lossless coder, arithmetic coding histograms mode (one of
      % LLC_AC_HST_xxxx)
      lossless_coder_AC_hist_type = CS_EncParams.LLC_AC_HST_MLTI;
    end
    
    methods
      function obj=CS_EncParams(def)
        if nargin > 0
          if isstruct(def)
            % If the matrix is SensingMatrixConvolve or SensingMatrixCnvlvRnd,
            % the default conv_mode should be SLCT_MODE_GLBL.
            if ~isfield(def,'conv_mode') &&...
                isfield(def, 'msrmnt_mtrx') && isfield(def.msrmnt_mtrx, 'type') && ...
                any(strcmp(def.msrmnt_mtrx.type, ...
                {'SensingMatrixConvolve', 'SensingMatrixCnvlvRnd', ...
                'SensingMatrixMLSeq', 'SensingMatrixMLSeqDC'}))
              def.conv_mode = SensingMatrixSqr.SLCT_MODE_GLBL;
            end
          end
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
            fls = fieldnames(v);
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
      
      function s = getStruct(obj)
        s = obj.getStruct@ProcessingParams();
        s.code = 0;
      end
    
      function mtrx = constructSensingMatrix(obj, vid_region, n_rows)
        % Construct a sesning matrix for the region specified by vid_region (an
        % object of class VidRegion.
        % The input argument n_rows is optional and relevant only to the case
        % that vid_region contains a single block. In that case it overrides the
        % computed number of measurements.
        
        n_blk = size(vid_region.blk_indx, 1);
        if n_blk > 1
          mtrcs = cell(n_blk, 1);
          for k=1:length(mtrcs)
            vdrg = VidRegion(vid_region.blk_indx(k,:), vid_region.blkr,...
              [obj.zero_ext_b; obj.zero_ext_f], obj.wrap_ext);
            mtrcs{k} = obj.constructSensingMatrix(vdrg);
          end
          mtrx = SensingMatrixBlkDiag.construct(mtrcs);
        else
          mt_args = obj.msrmnt_mtrx.args;
          if ~isfield(mt_args, 'prmt')
            mt_args.prmt = struct();
          end
          if ~isfield(mt_args.prmt, 'PL_mode')
            mt_args.prmt.PL_mode = obj.conv_mode;
          end
          mt_args.prmt.PL_range = obj.conv_rng;
          blk_indx = vid_region.blk_indx;
          mt_args.n_cols = vid_region.ext_vec_len;
          if nargin >= 3
            mt_args.n_rows = n_rows;
          else
            clr_pxls = vid_region.n_orig_blk_clr_pxls;
            for iclr = 1:size(vid_region.blkr.ovrlp,1)
              clr_pxls(iclr) = clr_pxls(iclr) * ...
                (1 - 2*vid_region.blkr.ovrlp(iclr,3)/vid_region.blkr.blk_size(iclr,3));
            end
          pxls = sum(clr_pxls);
          mt_args.n_rows = ceil(pxls * obj.msrmnt_input_ratio);
          end
          mt_args.rnd_seed =  obj.randomSeed(blk_indx, vid_region.blkr.blk_cnt);
          mt_args.prmt.PL_size = vid_region.blkr.ext_Cblk_size;
          mt_args.prmt.N_msrs = mt_args.n_rows;
      
          mtrx = SensingMatrix.construct(obj.msrmnt_mtrx.type, mt_args);
        end
      end
    end
    
    methods (Access=protected)
      function len = encodeCommon(obj, code_dst, flg_mtrx_args, ...
          flg_mtrx_nxt, flg_qntzr_exp_rng)
        % Encode parameters which are common to video and image
        len = 0;
        if obj.use_json
          len0 = code_dst.writeString(obj.getJSON());
          if ischar(len0);
            len = len0;
          else
            len = len+len0;
          end
          return;
        end;
        
        int_vals = [obj.random.seed  obj.conv_mode, obj.qntzr_wdth_mode, ...
          obj.lossless_coder_AC_hist_type];
        len0 = code_dst.writeUInt(int_vals);
        if ischar(len0); len = len0; return; end;
        len = len + len0;
        
        if isnumeric(obj.random.type)
          len0 = code_dst.writeString(show_str(obj.random.type));
        else
          len0 = code_dst.writeString(obj.random.type);
        end
        if ischar(len0); len = len0; return; end;
        len = len + len0;
        
        flt_vals = [obj.msrmnt_input_ratio, ...
          obj.qntzr_wdth_mltplr, obj.qntzr_ampl_stddev, ...
          obj.lossless_coder_AC_gaus_thrsh];
        if flg_qntzr_exp_rng
          flt_vals = [flt_vals, obj.qntzr_wdth_rng_exp];
        end
        len0 = code_dst.writeNumber(flt_vals);
        if ischar(len0); len = len0; return; end;
        len = len + len0;
        
        mtrx = obj.msrmnt_mtrx;
        len0 = code_dst.writeString(mtrx.type);
        if ischar(len0); len = len0; return; end;
        len = len + len0;
        if flg_mtrx_args
          json_mtrx = json_mat2strct(mtrx.args);
          json_mtrx = mat2json(json_mtrx, '', true);
          len0 = code_dst.writeString(json_mtrx);
          if ischar(len0); len = len0; return; end;
          len = len + len0;
        end
        if flg_mtrx_nxt
          json_nxt = json_mat2strct(mtrx.nxt);
          json_nxt = mat2json(json_nxt, '', true);
          len0 = code_dst.writeString(json_nxt);
          if ischar(len0); len = len0; return; end;
          len = len + len0;
        end
        
      end
      
      function len = decodeCommon(obj, code_src, cnt, flg_use_json, ...
        flg_mtrx_args, flg_mtrx_nxt, flg_qntzr_exp_rng)
        % Decode parameters which are common to video and image
        len = 0;
        if flg_use_json
          [jstr, n_rd, err_msg] = code_src.readString(cnt);
          if isnumeric(jstr)
            if jstr==-1
              len = 'EOD found';
            else
              len = err_msg;
            end
          else
            len = len + n_rd;
            def = ProcessingParams.parse_opts(jstr);
            obj.setParams(def);
          end
          return
        end
        
        [int_vals, n_rd] = code_src.readUInt(cnt, [1,4]);
        if ischar(int_vals) || (isscalar(int_vals) && int_vals == -1)
          len = int_vals;
          return;
        else
          len = len+n_rd;
          cnt = cnt - n_rd;
        end
        int_vals = double(int_vals);
        
        [rnd_type, n_rd, err_msg] = code_src.readString(cnt);
        if ~ischar(rnd_type)
          if rnd_type == 0
            len = err_msg;
          else
            len = 'Unexpected end of data';
          end
          return
        else
          len = len + n_rd;
          cnt = cnt - n_rd;
        end
        if regexp(rnd_type, '^\d+$')
          rnd_type = sscanf(rnd_type,'%d');
        end
        
        n_flt = 4;
        if flg_qntzr_exp_rng
          n_flt = n_flt + 1;
        end
        [flt_vals, n_rd] = code_src.readNumber(cnt, [1,n_flt]);
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
        
        [mtrx_type, n_rd, err_msg] = code_src.readString(cnt);
        if ~ischar(mtrx_type)
          if mtrx_type == 0
            len = err_msg;
          else
            len = 'Unexpected end of data';
          end
          return
        end
        len = len + n_rd;
        mtrx = struct('type', mtrx_type);
        if flg_mtrx_args
          [mtrx_args, n_rd, err_msg] = code_src.readString(cnt);
          if ~ischar(mtrx_args)
            if mtrx_args == 0
              len = err_msg;
            else
              len = 'Unexpected end of data';
            end
            return
          end
          mtrx_args = parse_json(mtrx_args);
          mtrx.args = json_strct2mat(mtrx_args);
          len = len + n_rd;
        else
          mtrx.args = struct();
        end
        if flg_mtrx_nxt
          [mtrx_nxt, n_rd, err_msg] = code_src.readString(cnt);
          if ~ischar(mtrx_nxt)
            if mtrx_nxt == 0
              len = err_msg;
            else
              len = 'Unexpected end of data';
            end
            return
          end
          mtrx_nxt = parse_json(mtrx_nxt);
          mtrx.nxt = json_strct2mat(mtrx_nxt);
          len = len + n_rd;
        else
          mtrx.args = [];
        end
        
        obj.random.type = rnd_type;
        obj.random.seed = int_vals(1);
        obj.conv_mode = int_vals(2);
        obj.qntzr_wdth_mode = int_vals(3);
        obj.msrmnt_input_ratio = flt_vals(1);
        obj.qntzr_wdth_mltplr = flt_vals(2);
        obj.qntzr_ampl_stddev = flt_vals(3);
        obj.lossless_coder_AC_gaus_thrsh = flt_vals(4);
        if flg_qntzr_exp_rng
          obj.qntzr_wdth_rng_exp = flt_vals(5);
        end
        obj.lossless_coder_AC_hist_type = int_vals(4);
        obj.msrmnt_mtrx = mtrx;
      end
    end
    
end

