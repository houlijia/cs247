% Documentation for this function is in VidRegion.m next to the function 
% signature
function len = encode(obj, code_dst, ~)
    len = code_dst.writeUInt(obj.n_blk);
    if ischar(len); return; end
    
    len1 = code_dst.writeUInt(obj.blk_indx(:));
    if ischar(len1);len = len1; return; end
    len = len + len1;
end

