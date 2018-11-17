function ofsts = next_frm_corr_ofsts(orig, step, rng_min, rng_max)
    % next_frm_corr_ofsts computes a set of offsets to be used in next frame
    % prediction. All input arguments are SimpleFractions objects and
    % represent row arrays of 2 entries (vertical, horizontal). The ourput
    % is an SimpleFractions of size [*,3].  In each row the first 2 entries
    % are horizontal and vertical entries (denoted as ofst) of the form
    %    ofst = orig + (i,j) .* step
    % such that
    %    rng_min <= ofst <= rng_max
    %    ofst ~= orig
    % The third entry in each row is always -1.
    %
    % Input:
    %   orig - origin
    %   step - step size
    %   rng_min - minimum range
    %   rng_max - maximmum range
    % Output:
    %   ofsts - a SimpleFractions of size [K,3], where each row is one
    %           offset. The first row corresponds to orig.
    
    low = fix((rng_min - orig) ./ step);
    high = fix((rng_max - orig) ./ step);
    sz = high - low + 1;
    indcs_v = (low(1):high(1))' * ones(1,sz(2));
    indcs_h = ones(sz(1),1) * (low(2):high(2));
    
    % Array of all offset indices
    indcs = [indcs_v(:), indcs_h(:)];
    
    % put origin first
    indcs(~any(indcs,2),:) = []; % remove origin
    indcs = [0,0; indcs]; % put it back in the beginning
    
    ofsts = ones(size(indcs,1),1) * orig;
    steps = ones(size(indcs,1),1) * step;
    ofsts = ofsts + indcs .* steps;
    ofsts = [ofsts, -ones(size(indcs,1),1)];
    ofsts = normalize(ofsts);
end

