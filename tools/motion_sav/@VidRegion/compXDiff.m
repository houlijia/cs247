% Documentation for this function is in VidRegion.m next to the function 
% signature
function [xcor, vec0_nrm] = compXDiff(obj, vec, offsets, nrm_exp)
    if iscell(vec)
        vec = obj.vectorize(vec);
    end
    if nargin < 4
        nrm_exp = 1;
    end
    
    vec_indcs = obj.inRngVec(offsets);
    vec_ofsts = obj.offsetPxlToVec(offsets, false);
    
    vec0 = vec(vec_indcs);
    vec0_nrm = norm(vec0, nrm_exp);
    
    xcor = zeros(size(vec_ofsts));
    for k=1:length(xcor)
        vec1 = vec(vec_indcs+vec_ofsts(k));
        xcor(k) = norm(vec0-vec1, nrm_exp);
    end
    xcor = 1 - xcor/(vec0_nrm + 1E-10);
end

