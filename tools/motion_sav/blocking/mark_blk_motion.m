function mrk_blk = mark_blk_motion(blk, mrkr_pos, clr, cnfdnc, vlcty)
    %mark_blk_motion - insert motion indicator into a block
    % Input:
    %   blk - a three dimensional array or a cell array of color components
    %   mrkr_pos - Marker position in the block [h,v], where h, are between
    %              -1 and 1.
    %   clr - array of 2 pixel values: The first is the color of the
    %         circle/ring indicating confidence.  The second is the color
    %         of the line segment showing direction.
    %   cnfdnc - confidence level (pair of numbers between 0 and 1)
    %   vlcty - Velocity [h,v] in units of block size fraction per frame
    %           (i.e. motion in pixels per frame divided by block size in
    %           that dimension.
    % 
    
    if iscell(blk)
        mrk_blk = blk;
        blk = mrk_blk{1};
        cell_inp = true;
    else
        cell_inp = false;
    end

    sz = size(blk);
    nfr = sz(3);
    
    % Map confidence to the [0,1] range
    cnfdnc = 0.5 * (cnfdnc + 1);
    cnfdnc = max(1E-10, min(1, cnfdnc));
    
    if ~any(vlcty)
        % Draw circle only in first frame of block
        for fr = 1:1
            blk(:,:,fr) = frm_draw_circle(blk(:,:,fr), clr(1), mrkr_pos,....
                cnfdnc(1)*0.25);
        end
    else
        vlcty = vlcty ./ sz(1:2);
        for fr = 1:nfr
            offset = vlcty * (fr - 0.5*nfr);
            cntr = mrkr_pos+offset;
            blk(:,:,fr) = frm_draw_ring(blk(:,:,fr), clr(1), cntr,....
                cnfdnc(1)*0.25, cnfdnc(2)*0.25);
            
            % Added velocity marker
            blk(:,:,fr) = frm_draw_line_seg(blk(:,:,fr), clr(2), ...
                cntr, (cntr+4*vlcty));
            blk(:,:,fr) = frm_draw_line_seg(blk(:,:,fr), clr(1), ...
                (cntr-2*vlcty), cntr);
        end
    end
    
    if cell_inp
        mrk_blk{1} = blk;
    else
        mrk_blk = blk;
    end

end

