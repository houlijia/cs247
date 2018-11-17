function [ origin, effective_size ] = calc_blk_position( blk_indx, vol_size, blk_size, ovrlp )
%calc_blk_position computes the position of a block within a video volume
%
%
%All input and output arguments are 1x3 arrays of [height, width,
%frames].
%
% INPUT arguments:
% blk_indx - index of the block
% vol_size - size of the volume from which the block is extracted
% blk_zie - the size of each block
% ovrlp - the overlap of adjacent blocks.
%
% OUTPUT arguments
% origin - the indices of the top-left-earliest pixel in the block
% effective_size - the last blocks in each dimension may be partial blocks,
%      where anything exceeding the boundary is filled with zeros.
%      effective_size gives the size which is actually a part of the
%      volume.

origin = (blk_indx - 1).*(blk_size - ovrlp)+1;
vls = ones(1,3);
vls(1:length(vol_size)) = vol_size;
effective_size = min(blk_size, (vls-(origin-1)));

end

