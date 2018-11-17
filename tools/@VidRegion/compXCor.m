% Documentation for this function is in VidRegion.m next to the function 
% signature
function [xcor, vec0_nrm, vec_nrms] = compXCor(obj, vec, vec_indcs, offsets)
    if iscell(vec)
        vec = obj.vectorize(vec);
    end
    
    vec_ofsts = obj.offsetPxlToVec(offsets, false);
    
    vec0 = vec(vec_indcs);
    vec0_nrm = norm(vec0);
    
    xcor = zeros(size(vec_ofsts));
    vec_nrms = xcor;

    for k=1:length(xcor)
        vec1 = vec(vec_indcs+vec_ofsts(k));
        vec_nrms(k) = norm(vec1);
        xcor(k) = dot(vec0, vec1);
    end
    xcor = xcor./(1E-10 + vec0_nrm*vec_nrms);
end
