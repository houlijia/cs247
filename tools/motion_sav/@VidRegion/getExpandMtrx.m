function [mtrx, intrp_mtrx] = getExpandMtrx(obj, level)
    % Compute a matrix which takes the pixel vector and expands it to
    % full block size
    % Input
    %   obj - this object
    %     level - level of expansion
    %       0 - No expansion (returned matrix is empty)
    %       1 - Only extension outside boundaries (intrp_mtrx is unit
    %           matrix
    %       2 - extension and interpolation
    % Output
    %   mtrx - the expansion matrix
    %   intrp_mtrx - interpolation only matrix
    
    if level == 0
        mtrx = [];
        intrp_mtrx = [];
        return
    end
    
    mtrcs = cell(obj.n_blk, obj.n_color);
    intrp_mtrcs = cell(obj.n_blk, obj.n_color);
    for iblk = 1:obj.n_blk
        [mt, imt] = obj.blkr.getExpandMtrx(obj.blk_indx(iblk,:),level);
        mtrcs(iblk, :) = mt;
        intrp_mtrcs(iblk,:) = imt;
    end
    
    if numel(mtrcs) > 1
        mtrx = SensingMatrixBlkDiag(mtrcs(:));
    else
        mtrx = mtrcs{1};
    end
    
    if nargout > 1
        if numel(intrp_mtrcs) > 1
            if level == 2
                intrp_mtrx = SensingMatrixBlkDiag(intrp_mtrcs(:));
            else
                % Just create a unit matrix
                intrp_mtrx = SensingMatrixUnit(mtrx.nCols);
            end
        else
            intrp_mtrx = intrp_mtrcs{1};
        end
    end                
end

