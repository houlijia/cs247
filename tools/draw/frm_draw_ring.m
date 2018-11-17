function frm = frm_draw_ring(frm,val,cntr,rds_out,rds_in,keep_aspect_ratio)
    %Input
    %  frm - a 2 dimensional array
    %  val - the value to set pixels to
    %  cntr - center of the circle (h,v), where the values are
    %        relative to the frame size, i.e. between 0 and 1.
    %  rds_out - outer radius  (in relative units 1 = one side of the block).
    %  rds_in - inner radius (same units)
    %  keep_aspect_ratio (optional) if true the larger dimension is reduced
    %           to keep the aspect ratio (default = true).
    
    if nargin < 6
        keep_aspect_ratio = true;
    end

    sz = size(frm);
    rd = circle_pxl_radius(sz, rds_out, keep_aspect_ratio);
    cntr = cntr.*sz;
    
    % rng is the range of a rectangle of pixels that completely contains
    % the ring. columns are [vertical; horizontal], rows are [begin, end].
    rng = [floor(cntr - rd); ceil(cntr + rd)];
    if any(rng(2,:) < [1,1]) || any(rng(1,:)  > sz)
        return;  % Completely out of the range of the picture
    end
    rng_len = rng(2,:) - (rng(1,:) - 1);
    cntr = cntr - (rng(1,:) - 1);
    f = false(rng_len);
    
    f = frm_draw_circle(f,true,cntr,rds_out,keep_aspect_ratio, true);
    rad_ratio = rds_in/rds_out;
    pcntr = cntr./rng_len;
    pt = find(f(:));
    pts = zeros(length(pt),2);
    [pts(:,1), pts(:,2)] = ind2sub(rng_len, pt);
    pts = pts./(ones(size(pts,1),1)*rng_len);
    for k=1: size(pts,1)
        strt = pcntr + rad_ratio * (pts(k,:)-pcntr);
        f = frm_draw_line_seg(f, true, strt, pts(k,:));
    end
        
    [v,h] = ind2sub(rng_len, find(f(:)));
    pts = [v,h] + ones(size(v,1),1)*(rng(1,:)-1);
    pts = draw_remove_outliers(pts,sz);
    frm(sub2ind(sz,pts(:,1),pts(:,2))) = val;
    
end

