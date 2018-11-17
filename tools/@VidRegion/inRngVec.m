% Documentation for this function is in VidRegion.m next to the function 
% signature
function vec_indcs = inRngVec(obj, pxl_ofst, blk_stt)
    % Create a region of all zeros
    pxls = obj.pixelize(zeros(obj.stt_vec_len(blk_stt),1), blk_stt);
    
    bgn_ofst = ones(1,3);
    bgn_ofst = max(bgn_ofst, bgn_ofst - min(pxl_ofst));
    end_ofst = zeros(1,3);
    end_ofst = max(end_ofst, max(pxl_ofst));
    
    for i_clr = 1:obj.n_color
        for i_blk = 1:obj.n_blk
            
            blk = pxls{i_clr, i_blk};
            blk(bgn_ofst(1):end-end_ofst(1),...
                bgn_ofst(2):end-end_ofst(2),...
                bgn_ofst(3):end-end_ofst(3)) = 1;
            pxls{i_clr, i_blk} = blk;
        end
    end
    vec_indcs = find(obj.vectorize(pxls));
end
