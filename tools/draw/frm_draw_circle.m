function frm = frm_draw_circle(frm,val,cntr,rds,keep_aspect_ratio, pxl_cntr)
    %Input
    %  frm - a 2 dimensional array
    %  val - the value to set pixels to
    %  cntr - center of the circle (h,v), where the values are
    %        relative to the frame size, i.e. between 0 and 1.
    %  rds - Radius  (in relative units 1 = one side of the block).
    %  keep_aspect_ratio - (optional) if true the larger dimension is reduced
    %           to keep the aspect ratio (default = true).
    %  pxl_cntr - (optional) if true center is given in pixel units, rather
    %             than relative units (default = false)
    
    if nargin < 6
        pxl_cntr = false;
        if nargin < 5
            keep_aspect_ratio = true;
        end
    end
    sz = size(frm);
    
    if ~pxl_cntr
        cntr = sz.*cntr;
    end
    
    rds = circle_pxl_radius(sz, rds, keep_aspect_ratio);
    fcrcl = @(t) ([cntr(1)+rds*cos(t), cntr(2)+rds*sin(t)]);
    
    pts = fcrcl((0:(0.2/sum(sz)):(2*pi))');
    pts = round(pts);
    pts = draw_remove_outliers(pts,sz);
    ind = sub2ind(sz,pts(:,1),pts(:,2));
    ind = unique(ind);
    
    frm(ind) = val;
end

