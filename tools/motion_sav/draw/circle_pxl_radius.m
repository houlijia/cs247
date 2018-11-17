function rds = circle_pxl_radius(sz, rds, keep_aspect_ratio)
    % circle_pxl_radius compute radius of a circle in pixels.
    %   Input:
    %     sz - size of frame (vertical, horizontal)
    %     rds - Radius  (in relative units 1 = one side of the block)
    %     keep_aspect_ratio if true the larger dimension is reduced
    %           to keep the aspect ratio.
    %   Output
    %     rds - Radius in pixels (may be fractional).  If
    %     keep_aspect_ratio is true it is a scalar. Otherwise it is an
    %     array of radiuses (horizontal, vertical)
    
    
    if keep_aspect_ratio
        rds = min(sz(1),sz(2))*rds;
    else
        rds = sz .* rds;
    end

end

