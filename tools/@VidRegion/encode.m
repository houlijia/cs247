% Documentation for this function is in VidRegion.m next to the function 
% signature
function len = encode(obj, code_dst, ~)
  len = code_dst.writeUInt(uint8(obj.whole_frames));
  if ischar(len); return; end
  
  if obj.whole_frames
    len1 = code_dst.writeUInt([obj.blk_indx(1,3)-1,...
      obj.blk_indx(end,3)-obj.blk_indx(1,3)]);
    if ischar(len1); len=len1; return; end 
    len = len+len1;
  else
    len1 = code_dst.writeUInt(obj.n_blk);
    if ischar(len1); len=len1; return; end 
    len = len+len1;
    
    len1 = code_dst.writeUInt(obj.blk_indx(:));
    if ischar(len1);len = len1; return; end
    len = len + len1;
  end
end

