function enc_opts = getImageEncOptsStruct(enc_opts )
  %getImageEncOpts computes a full struct corresponding to CS_ImgEncParams object.
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
  %    enc_opts - a struct derived form CS_EncImgParams object
  %          
  
  if nargin < 1
    enc_opts = getImageEncOpts();
  else
    enc_opts = getImageEncOpts(enc_opts);
  end
  
  enc_opts = enc_opts.getStruct();
end

