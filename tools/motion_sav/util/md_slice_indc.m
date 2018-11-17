function slc_indcs = md_slice_indc( sz, dm, indc )
    %md_slice_indc computes the linear indices of a slice in a
    %multi-dimensional array.  A slice in a MD array along a specific
    %dimension dm is a the set of all entries for which the dm-th index has
    %specific values.
    %  Input
    %    sz - the size of the MD array (row vector)
    %    dm - the dimension on which the slice is taken
    %    indc - the specific indices in for the slice (scalar of vector).
    %  Output
    %    slc_indcs - the slice linear indices, as a column vector. If indc 
    %    entries are in increasing order so are the entries of slc_indcs.
    
    l_pre = prod(sz(1:(dm-1)));
    l_dm = l_pre * sz(dm);
    l_post = prod(sz((dm+1):end));
    
    if ~isrow(indc)
        indc = indc';
    end
    
    slc_indcs = (1:l_pre)';
    slc_indcs = slc_indcs * ones(1,length(indc)) + ...
        ones(length(slc_indcs),1)*(l_pre*(indc-1));
    slc_indcs = slc_indcs(:);
    slc_indcs = slc_indcs * ones(1,l_post) + ...
        (l_dm*ones(length(slc_indcs),1))*(0:(l_post-1));
    slc_indcs = slc_indcs(:);
end

