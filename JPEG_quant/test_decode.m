function [rec_q, qy, sat_ind, qm, coding_info, rec_q_raw] =...
    test_decode(cs_file, dec_opts, rec_q_file, ref)
  % test_decode gets a CS compressed image, decodes the lossless codign,
  % unquantizes and reconstructs the original image.
  % 
  % INPUT
  %   cs_file: If it is a byte array (uint8) this is the coded image. If it is a
  %       character string, it is the name of a file containing the data.
  %   dec_opts: (optional) Should be a struct or a cell array of structs, where
  %             each struct specifies a reconstruction algorithm. Each struct
  %             should have at least the field 'alg', which species the
  %             algorithm. Other algorithm parameters may be specified as well.
  %             If not present or empty, set to struct('alg', 'GAP_TV').
  %   rec_q_file: (optional) If present and not empty, output file for 
  %               reconstructed image(s).  If dec_opts is a struct or not
  %               present, this should be a string with the file specifications.
  %               If dec_opts is a cell array, req_q_file should be a cell array
  %               of the same size, containing file names for each reconstructed
  %               image.
  %       from quantized measurements.
  %   ref: An optional argument for debugging. If present and not zero it
  %         should be a struct containing enc_opts and qm generated in
  %         quantization and it verifies that they are identical to the outputs
  %         coding_info.enc_opts and qm, respectively.
  % OUTPUT
  %   rec_q: If dec_opts is a struct or not present, this is an array containing 
  %          the reconstructed image. Otherwise it is a cell array, with each
  %          cell containing the image reconstructed according to the
  %          corresponding dec_opts.
  %   qy: Quantized measurements
  %   sat_ind: Indices of saturated measurements
  %   qm: QuantMeasurements objects containing the measurements and related
  %       information.
  %   coding_info - coding info read from the compressed file
  %   rec_q_raw - a cell array containing the raw images (as produced by
  %               reconstruction, before any normalization).
  
  % Initialization for MEX
  mex_clnp = initMexContext();   %#ok (prevent error on unused variable)
  
  if nargin < 2 || isempty(dec_opts)
    dec_opts = struct('alg', 'GAP_TV');
  end
  if nargin < 3
    rec_q_file = '';
  end
  if nargin < 4
    ref = struct();
  end
  
  % Decode
  [qy, coding_info, sat_ind, qm] = CSUnquantize(cs_file, ref);
  
  % Reconstruct from quantized measurements
  [rec_q, rec_q_raw] = CSImgReconstruct(qy, coding_info, dec_opts);
  if ~isempty(rec_q_file)
    if ~iscell(dec_opts)
      rec_q = {rec_q};
      rec_q_raw = {rec_q_raw};
      rec_q_file = {req_q_file};
    end
    for k=1:numel(dec_opts)
      q_img = rec_q{k} / max([rec_q{k}(:);1E-10]);
      if isa(q_img, 'gpuArray')
        q_img = gather(q_img);
      end
      imwrite(q_img, rec_q_file{k});
      rec_q_raw_file = regexprep(rec_q_file{k},'[.][^.]*$', '-raw_rec.mat');
      raw_rec_img = rec_q_raw{k};   %#ok
      save(rec_q_raw_file, 'raw_rec_img');
    end
    if ~iscell(dec_opts)
      rec_q = rec_q{1};
    end
  end
  
end
