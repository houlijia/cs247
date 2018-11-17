function pts = draw_remove_outliers(pts, sz)
    %draw_remove_outliers - remove points which are out of range from a
    %list of points
    % Input:
    %   pts - a list of points (pixel position). It is a 2 dimensional
    %         array where each point is a row. The dimension d of the space
    %         is the number of entries in a row.
    %   sz  - frame size. A row of the same size as the pts rows (not
    %         necessarily 2)
    % Output
    %   pts - the input pts after removal of all points outside the range
    %         of sz
    
    for d=1:length(sz)
        pts(pts(:,d)<1 | pts(:,d)>sz(d),:) = [];
    end
    
end

