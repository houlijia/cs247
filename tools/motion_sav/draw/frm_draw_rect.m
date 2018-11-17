function frm = frm_draw_rect(frm,val,pos,lwdth,keep_aspect_ratio)
    %frm_draw_rect Draw a rectangle in a 2 dimensional frame (array)
    %  
    %Input
    %  frm - a 2 dimensional array
    %  val - the value to set pixels to
    %  pos - position of the rectangle in the frames.  This is a 2x2 array
    %        where the first row is the beginning point of (h,v)
    %        (horizontal, vertical  or 1,2) axes and the values are
    %        relative to the frame size, i.e. between 0 and 1.
    %  lwdth - (optional)line width - a 2x2 array of line widths relative
    %           to the rectangle size. first colomn is horizontal widths
    %           and 2nd is vertical widths. The defalut width is 0, meaning
    %           one pixel
    %  keep_aspect_ratio (optional) if true the larger dimension is reduced
    %           to keep the aspect ratio.
    
    if nargin < 5
        keep_aspect_ratio = false;
        if nargin < 4
            lwdth = zeros(2,2);
        end
    end
    
    if keep_aspect_ratio
        if size(frm,1) > size(frm,2)
            ratio = size(frm,2)/size(frm,1);
            pos(:,1) = 0.5 + (pos(:,1)-0.5)*ratio;
            lwdth(:,1) = lwdth(:,1) * ratio;
        elseif size(frm,1) < size(frm,2)
            ratio = size(frm,1)/size(frm,2);
            pos(:,2) = 0.5 + (pos(:,2)-0.5)*ratio;
            lwdth(:,2) = lwdth(:,2) * ratio;
        end
    end
    
    % outside of rectangle, in pixels
    pos_out = ceil(pos.*[size(frm);size(frm)]);
    pos_out = max(ones(2,2),pos_out);
    
    % Inside of rectangle in pixels;
    pos_in = pos;
    pos_in(1,:) = pos_in(1,:) + lwdth(1,:);
    pos_in(2,:) = pos_in(2,:) - lwdth(2,:);
    pos_in = ceil(pos_in.*[size(frm);size(frm)]);
    pos_in = max(ones(2,2),pos_in);
    
    % Draw vertical lines
    frm = pix_blk_draw(frm,val, pos_out(1,1):pos_out(2,1), ...
        [pos_out(1,2):pos_in(1,2) pos_in(2,2):pos_out(2,2)], 1);
    
    % Draw horizontal lines
    frm = pix_blk_draw(frm,val, ...
        [pos_out(1,1):pos_in(1,1) pos_in(2,1):pos_out(2,1)],...
        pos_in(1,2):pos_in(2,2), 1);
end


