function frm = frm_draw_line_seg(frm, val, pstart, pend)
    %frm_draw_line_seg
    %   Detailed explanation goes here
    
 sz = size(frm);
 pstart = sz.*pstart;
 pend = sz.*pend;
 df = pend - pstart;
 n = ceil(max(abs(df)))+1;
 df = df / n;
 pts = ones(n+1,1)*pstart + (0:n)'*df;
 pts = round(pts);
 pts = draw_remove_outliers(pts,sz);
 ind = sub2ind(sz, pts(:,1), pts(:,2));
 frm(ind) = val;
end

