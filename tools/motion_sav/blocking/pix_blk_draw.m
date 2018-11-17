function blk = pix_blk_draw(blk, val, v, h, t, inv)
    %pix_blk_draw sets pixel values in specific positions in the blocks to
    %a specficn values
    %Input
    %  blk - a three dimensional array
    %  val - the value to set
    %  v - vertical coordinates (1st index)
    %  h - horizontal coordinates (2nd index)
    %  t - temporal coordinates (3rd index)
    %  inv - (optional) an array of 3 logical values. For each entry, if
    %        true the corresponding index is inverted (see more below).
    %Output
    %  blk - input blk after modification
    %
    %h,v,t may be a single value or an array of coordintates. Non-positive
    %coordinates are translated to offsets from the end of the block.  The
    %function will set blk(i,j,k) to val for all i,j,k in h,v,t, 
    %respectively.  If inv is set for a particular index, valn is set for
    %all i,j,k which are NOT in the corresponding h,v,t, respectively.
    
    v(v<=0) = size(blk,1)+v(v<=0);
    h(h<=0) = size(blk,2)+h(h<=0);
    t(t<=0) = size(blk,3)+t(t<=0);
    
    if nargin >= 6
        for k=1:3
            if ~inv(k)
                continue;
            end
            f = 1:size(blk,k);
            switch k
                case 1
                    f(v)=[];
                    v=f;
                case 2
                    f(h)=[];
                    h=f;
                case 3
                    f(t)=[];
                    t=f;
            end
        end
    end
    
    blk(v,h,t) = val;
    
end

