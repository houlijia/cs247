function ofsts = get_neighbor_ofsts(edge_range)
% Compute the offsets to a pixel neighbors
% Input:
%     edge_rng - number of pixels (V,H,T) to be used on both sides for edge
%                detection
% Output:
%     ofsts - An array of offsets to neighbors (*,3). The first offset is 
%             [0,0,0] (the pixel itself).

    erng = zeros(1,3);
    erng(1:length(edge_range)) = edge_range;
    ofsts = zeros(prod(2*erng+1)-1, 3);
    entry_indx = 1;
    for ii=-erng(1):erng(1)
        for jj=-erng(2):erng(2)
            for kk=-erng(3):erng(3)
                if ~ii && ~jj && ~kk
                    continue;
                end
                ofsts(entry_indx,:) = [ii,jj,kk];
                entry_indx = entry_indx + 1;
            end
        end
    end
    ofsts = [0,0,0; ofsts];
end