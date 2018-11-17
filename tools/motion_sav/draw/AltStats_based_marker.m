function  marker = AltStats_based_marker(blk, mrkr_pos, clr, activity_stat, thrhld)
    
%thrhld= 0.5;

    if iscell(blk)
        marker = blk;
        blk = marker{1};
        cell_inp = true;
    else
        cell_inp = false;
    end
    
     sz = size(blk);
     nfr = sz(3);
     
     %activity_stat = dec_data.activity_statistic;
     
if  activity_stat < thrhld
    % Draw circle only in the first frame of block
    for fr = 1:1
            %blk(:,:,fr) = frm_draw_circle(blk(:,:,fr), clr(1), mrkr_pos,....
              %  thrhld/15);
             %blk(:,:,fr) = frm_draw_circle(blk(:,:,fr), clr(2), mrkr_pos,....
               % activity_stat/15);
             blk(:,:,fr) = frm_draw_line_seg(blk(:,:,fr), clr(2), ...
                mrkr_pos, (mrkr_pos+activity_stat/300));
            blk(:,:,fr) = frm_draw_line_seg(blk(:,:,fr), clr(1), ...
                (mrkr_pos-thrhld/300), mrkr_pos);
    end
else % Draw ring (two circles)
        for fr = 1:nfr
           % blk(:,:,fr) = frm_draw_circle(blk(:,:,fr), clr(1), mrkr_pos,...
            %    thrhld/15);
            %blk(:,:,fr) = frm_draw_circle(blk(:,:,fr), clr(2),mrkr_pos,...
             %   min(1,activity_stat/15));
            
            % Added marker
            blk(:,:,fr) = frm_draw_line_seg(blk(:,:,fr), clr(2), ...
                mrkr_pos, (mrkr_pos + activity_stat/300));
            blk(:,:,fr) = frm_draw_line_seg(blk(:,:,fr), clr(1), ...
                (mrkr_pos - thrhld/300), mrkr_pos);
        end
end

if cell_inp
        marker{1} = blk;
else
        marker = blk;
end

end
        