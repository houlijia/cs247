function [orig_img, y, qm, q_code, coding_info] = test_encode(...
    image_file, enc_opts, cs_file)
  % test_encode computes measurements of an image, then quantizes and codes
  % them.
  % 
  % INPUT
  %   image_file: Input image file name
  %   enc_opts - (optional): either a CS_EncImgParams object or a struct
  %     containing options which measurements generation encoding and decoding.
  %     Any field which is not specified gets a default value, specified in the
  %     struct def_opts in getImageCodingInfo(). 
  %   cs_file: If present and not empty, output file for compressive measurements.
  % OUTPUT
  %   orig_img: Original image as a floating point array
  %   y: Unquantized measurements
  %   qm: QuantMeasurements objects containing the measurements and related
  %       information.
  %   q_code: A byte array (uint8) containing the coded image (content of
  %       cs_file.
  %   coding_info - a struct which cotains all the parameters necessary to
  %      compute, encode, decode and reconstruct the measurements. For details
  %      see compCodingInfo
  
  % Initialization for MEX
  mex_clnp = initMexContext();   %#ok (prevent error on unused variable)
  
  % Parse input arguments
  if nargin < 2
    enc_opts = struct();
  end
  if nargin < 3
    cs_file = '';
  end
  
  % Read pixels
  img_info = imfinfo(image_file);
  orig_img=imread(image_file);
  
  % Set parameters for quantization
  coding_info = getImageCodingInfo(img_info, enc_opts);
  
  % Remove color if necessary
  if ~coding_info.enc_opts.process_color && size(orig_img,3) > 1
    orig_img = rgb2gray(orig_img);
  end
  
  % Convert source image to right type
  orig_img = coding_info.sens_mtrx.toFloat(im2double(orig_img));
      
  % Compute measurements
  y = coding_info.sens_mtrx.multVec(orig_img(:));
  
  % Quantize and code the measurements
  [q_code, qm] = CSQuantize(y, coding_info, cs_file);
  
  
end
