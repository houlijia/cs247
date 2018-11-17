classdef CSVideoBlockProcessingData < handle
% contains data on how a video block was processed by the 
% CompressedSensedVideoCodec. It also contain the original video block and 
% the reconstructed video block.    
   
    properties
   
        % Block vertical, horizontal and temporal indices
        blk_indx;
      
        % No. of pixels in block
        n_pxl;
       
        % No. of measruements
        n_blk_msrmnts;
       
        % DC of input
        dc_value_of_input=-1e-10;  % indicates unspecified yet
        
        % Mean and variance of measurements without DC
        mean_msrmnts_nodc;
        variance_msrmnts_nodc;
       
        n_msrmnts_outside;
       
        n_msrmnts_no_clip;
       
       % Number of bits used to encode the block
       n_bits;
       
       % Number of bits used for quantized measurements
       n_msrmnts_bits;
       
       % Number of enropy bits per measurement
       n_entrp_per_msrmnt;
       
       % Number of bits per measurement spent to indicate if it is 
       % clipped or not
       n_entrp_clipped;
       
       n_qntzr_bins;
       
       seconds_to_process_block;
                               
   end %properties
   
   methods
       
       function obj = CSVideoBlockProcessingData(indx,...
               info, blk_len_bytes, msrs_len_bytes, dc_val)
           if nargin >= 1
               obj.blk_indx = indx;
               if nargin >= 2
                   obj.n_pxl = info.vid_region.nPxlInRegion();
                   obj.mean_msrmnts_nodc = info.q_msr.mean_msr;
                   obj.variance_msrmnts_nodc =info.q_msr.stdv_msr^2;
                   obj.n_qntzr_bins=...
                       info.q_msr.n_bins;
                   obj.n_msrmnts_outside=...
                       length(find(info.q_msr.bin_numbers >= info.q_msr.n_bins)) +...
                       length(find(info.q_msr.bin_numbers <= 0));

                   obj.n_blk_msrmnts = length(info.q_msr.bin_numbers);
                   obj.n_msrmnts_no_clip = info.q_msr.n_no_clip;
                   obj.n_msrmnts_bits = msrs_len_bytes*8;
                       
                   [entrp, used_prob] = ...
                       info.q_msr.compEntropy();
                   obj.n_entrp_per_msrmnt = entrp /log(2);
                   if used_prob < 1;
                       actual_used_prob = 1 - ...
                           obj.n_msrmnts_outside/obj.n_blk_msrmnts;
                       obj.n_entrp_clipped = -(...
                           actual_used_prob * log2(used_prob) +...
                           (1-actual_used_prob) * log2(1-used_prob));
                   else
                       obj.n_entrp_clipped = 0;
                   end

                   if nargin >= 3
                       obj.n_bits = blk_len_bytes * 8;
                       
                       if nargin >= 4
                           obj.dc_value_of_input=dc_val;
                       end
                   end
               end
           end
       end
           
   end %methods
    
end