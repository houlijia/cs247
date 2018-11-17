function ofst_vec = comp_pxl_ofsts_vec(pxl_vec, ofsts)
%comp_pxl_ofsts_vec computes an array of shifted versions of a pixel vector
%   Input: 
%      pxl_vec - a pixel vector
%      ofsts - a vector of integer offsets
%   output:
%      ofst_vec - an array. The first column is pxl_vec. The following
%                 columns are pxl_vec shifted (circularly) by the offests
%                 in ofst

    ofst_vec = zeros(length(pxl_vec), length(ofsts)+1);
    ofst_vec(:,1) = pxl_vec(:);
    for k=1:length(ofsts)
        ofst = mod(ofsts(k)-1, length(pxl_vec))+1;
        ofst_vec(:,k+1) = [pxl_vec(1+ofst:end); pxl_vec(1:ofst)];
    end

end

