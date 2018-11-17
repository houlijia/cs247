function enc_opts = getImageEncOpts(enc_opts )
  %getImageEncOpts computes a CS_ImgEncParams object.
  %  INPUT:
  %     enc_opts - (optional): either a CS_EncImgParams object or a struct
  %       containing options which control encoding and decoding. Any field 
  %       which is not specified gets a default value, specified in the
  %       struct def_opts in this function. Use it as an example for
  %       reasonable values. The fields can be:
  %         process_color: If true and the image has color, process it as a
  %           color image. Otherwise, encode only the luminance.
  %         qntzr_wdth_mode: Determines how the next parameter,
  %           qntzr_wdth_mltplr, is interpreted:
  %             CS_EncParams.Q_WDTH_ABS: qntzr_wdth_mltplr is the actual
  %               quantizer step size.
  %             CS_EncParams.Q_WDTH_NRML: qntzr_wdth_mltplr is 
  %               (quantizer step size / noise std.dev)
  %             CS_EncParams.Q_WDTH_CSR: qntzr_wdth_mltplr is 
  %               (quantizer step / noise std.dev) * measurement_input_ratio
  %         qntzr_wdth_mltplr: Normalized quantization step. The quantization
  %           step is this value * standard deviation of the measurements noise.
  %         qntzr_ampl_stddev: Normalized quantizer amplitude. The quantizer 
  %           amplitude is the range beyond which measurements are saturated. 
  %           The amplitude is set to this value times the standard deviation of
  %           the measurements.
  %         qntzr_outrange_action: Can be:
  %           CS_EncImgParams.Q_DISCARD: Discard saturated measurements.
  %           CS_EncImgParams.Q_SAVE: Save the unsaturated value of saturated
  %             measurements.
  %         use_single: If true use single precision in quantization and coding.
  %         lossless_coder: can be:
  %           CS_EncImgParams.LLC_AC: Use arithmetic coding
  %           CS_EncImgParams.LLC_INT: Use one or more byres for each measurement.
  %         msrmnt_input_ratio: required ratio between number of measurements
  %           and number of pixels in the image
  %         CSr: Same as measurement_input_ratio
  %         msrment_mtrx: Specifies the sensing matrix. Can be either
  %           struct('type','SensingMatrixWH') - Walsh Hadamard with input
  %             permutation
  %           struct('type','SensingMatrixNrWH') - Walsh Hadamard without input
  %             permutation
  %         do_permut: Can be used instead of msrmnt_mtrx. If true there is
  %            input permutation ('SensingMatrixWH'), otherwise there is not 
  %            ('SensingMatrixNrWH').
  %  OUTPUT:
  %    enc_opts - a CS_EncImgParams object
  %          
  
  def_opts = struct(...
    'process_color', false, ...
    'qntzr_wdth_mode', CS_EncParams.Q_WDTH_NRML, ...
    'qntzr_wdth_mltplr', 10,...  
    'qntzr_ampl_stddev', 4,...  
    'qntzr_outrange_action', CS_EncImgParams.Q_SAVE,...
    'use_single', true, ... 
    'lossless_coder', CS_EncImgParams.LLC_AC, ... 
    'msrmnt_input_ratio', 0.2, ...
    'msrmnt_mtrx', struct('type','SensingMatrixNrWH') ...
    );
    
  if nargin < 1
    enc_opts = CS_EncImgParams(def_opts);
  elseif isstruct(enc_opts)
    if isfield(enc_opts, 'CSr')
      if ~isfield(enc_opts, 'msrmnt_input_ratio')
        enc_opts.msrmnt_input_ratio = enc_opts.CSr;
        enc_opts = rmfield(enc_opts, 'CSr');
      else
        error('test_quant:input', ...
          'cannot specify both CSr and msrmnt_input_ratio in enc_opts');
      end
    end
    
    if isfield(enc_opts, 'do_permut')
      if ~isfield(enc_opts, 'msrmnt_mtrx')
        if enc_opts.do_permut
          enc_opts.msrmnt_mtrx = struct('type','SensingMatrixWH');
        else
          enc_opts.msrmnt_mtrx = struct('type','SensingMatrixNrWH');
        end
        enc_opts = rmfield(enc_opts, 'do_permut');
      else
        error('test_quant:input', ...
          'cannot specify both do_permut and msrmnt_mtrx in enc_opts');
      end
    end
    flds = fieldnames(def_opts);
    for k=1:length(flds)
      fld = flds{k};
      if ~isfield(enc_opts, fld)
        enc_opts.(fld) = def_opts.(fld);
      end
    end
    enc_opts = CS_EncImgParams(enc_opts);
  end
end