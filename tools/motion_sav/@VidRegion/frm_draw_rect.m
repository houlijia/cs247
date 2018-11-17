function vec = frm_draw_rect(obj,vec,frm_no,clr,val,pos,lwdth,keep_aspect_ratio)
    %frm_draw_rect See in VidRegion.m
    
    if ~iscell(vec)
        vec = obj.pixelize(vec);
        vec_is_vector = true;
    else
        vec_is_vector = false;
    end
    
    args = cell(1,nargin-3);
    args{3} = pos;
    if nargin >= 7
        args{4} = lwdth;
        if nargin >=8
            args{5} = keep_aspect_ratio;
        end
    end
    
    if isscalar(val)
        val = ones(size(clr))*val;
    end
    
    for iclr=clr
        args{2} = val(iclr);
        for iblk=1:size(obj.blk_indx,1)
            blk = vec{iclr,iblk};
            for f_no = frm_no
                args{1} = blk(:,:,f_no);
                blk(:,:,f_no) = frm_draw_rect(args{:});
            end
            vec{iclr,iblk} = blk;
        end
    end
    
    if vec_is_vector
        vec = obj.vectorize(vec);
    end
end

