function mtstr = mtrx_to_str(mtrx)
    if all(abs(mtrx(:))) < 10
        if min(mtrx) >= 0
            fmt0 = '%5.3f';
        else
            fmt0 = '%+6.3f';
        end
    elseif min(mtrx) >= 0
        fmt0 = '%7.3e';
    else
        fmt0 = '%+8.3e';
    end
    
    fmtc = { ['%s' fmt0], ['%s ' fmt0];...
        ['%s' fmt0 '\n'], ['%s ' fmt0 '\n']};
    
    mtstr = '';
    for i=1:size(mtrx,1)
        for j=1:size(mtrx,2)
            fmt = fmtc{1+(j == size(mtrx,2)), 1+(j >1)};
            mtstr = sprintf(fmt, mtstr, mtrx(i,j));
        end
    end
end
