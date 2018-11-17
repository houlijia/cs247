function frm = frm_draw_fill(frm,val)
    %frm_draw_fill Fill the value val inside the shape drawn by the
    %non-zero elements of frm.
    
    b = zeros(size(frm)+[2,2]);
    b (2:end-1,2:end-1) = (frm ~= 0);  % Non zero entries in frm
    
    % c is the decision about pixels in frm. Can get the following values:
    % 0  - not decided yet
    % 1  - set entries in b which are outside boundary (to be filled inside)
    % -1 - set entries in b which are inside boundary (e.g. inner circle in to be filled
    %       a ring) and need to be filled outside
    % 2  - unset entries in b which are to be filled
    % -2 - unset entries in b which are not to be filled (outside points
    c = zeros(size(b));

    function find_rect()
        d = ~c(v_bgn:v_end, h_bgn:h_end);
        h_or = sum(d,1); % or along vertical dimension
        if ~any(h_or)
            h_bgn = 0; v_bgn = 0;
            return
        end
        h_end = h_bgn-1 + find(h_or,1,'last');
        h_bgn = h_bgn-1 + find(h_or,1,'first');
        
        v_or = sum(d,2); % or along vertical dimension
        v_end = v_bgn-1 + find(v_or,1,'last');
        v_bgn = v_bgn-1 + find(v_or,1,'first');
    end

    v_bgn=2; h_bgn = 2;
    v_end=size(b,1)-1; h_end = size(b,2)-1;
    if ~v_bgn
        return;
    end
    
    % Fill boundaries and outside boundaries
    c(:) = -2;  % Initially everything is outside
    c(v_bgn,(b(v_bgn,:)==1)) = 1;
    c(v_end,(b(v_end,:)==1)) = 1;
    c((b(:,h_bgn)==1),h_bgn) = 1;
    c((b(:,h_end)==1),h_end) = 1;
    c(v_bgn+1:v_end-1,h_bgn+1:h_end-1) = 0;  % Rect. of not yet determined pixels
    
    % start by trying to expand the ouside area
    if any(any(c(v_bgn:v_end,h_bgn:h_end) == -2))
        btype = 0;
        ctype = -2;
        c(c==1) = 0;
    else
        btype = 1;
        ctype = 1;
    end
    
    while v_bgn
        while(v_bgn)
            d = ~c(v_bgn:v_end,h_bgn:h_end) & ...
                b(v_bgn:v_end,h_bgn:h_end) == btype &(...
                c(v_bgn+0:v_end+0,h_bgn+1:h_end+1) |...
                c(v_bgn+1:v_end+1,h_bgn+0:h_end+0) |...
                c(v_bgn-1:v_end-1,h_bgn+0:h_end+0) |...
                c(v_bgn+0:v_end+0,h_bgn-1:h_end-1) ...
                );
            
            if ~any(d(:))
                break;
            end
            d0 = zeros(size(c));
            d0(v_bgn:v_end,h_bgn:h_end) = d;
            c(d0~=0) = ctype;
            
            find_rect();
        end
            
        switch ctype
            case 1  % ctype==1 is ouside boudary
                ctype = 2; btype = 0;
            case 2  % ctype==2 is inside fill
                ctype = -1; btype = 1;
            case -1 % ctype==-1 is inside boundary (e.g. ring)
                ctype = -2; btype = 0;
            case -2 % ctype==-2 is outside fill (e.g in inner circle of ring)
                ctype = 1; btype = 1;
        end
    end
    
    c = c(2:end-1,2:end-1);
    frm((c==2)) = val;
end

