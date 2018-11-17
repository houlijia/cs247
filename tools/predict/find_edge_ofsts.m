function [needed_ofsts, ofsts_list, nghbr_list] = ...
    find_edge_ofsts(nghbr_ofsts, ofsts)
  %  Compute the offsets needed to perform edge detection.
  %  Input
  %     nghbr_ofsts  - An array of offsets to neighbors (M+1,3). The first offset is
  %             [0,0,0] (the pixel itself).
  %     ofsts - An offsets matrix of size (K,3), where each row is a desired
  %             (V,H,T) offset
  %  Output
  %     needed_ofsts - a (L,3) matrix of offsets, which contains ofsts as
  %                    well as all the additional offsets needed for edge
  %                    detection.
  %     ofsts_list - a vector of indices into needed_ofsts which points to
  %                  the original ofsts table:
  %                     needed_ofsts(ofsts_list(k),:) == ofsts(k,:)
  %     nghbr_list - an array of indices of size (K,M) into needed_ofsts,
  %                  which points to the neighbors of offsets in ofsts. Thus
  %                  needed_ofsts(nghbr_list(k,:),:) are the neighbors of
  %                  ofsts(k,:).

  [ofsts, denom] = rat(ofsts); % work only with nominator
  K = size(ofsts,1);
  L = size(nghbr_ofsts,1);
  n_ofsts = zeros(K*L, 3);
  
  for j=1:3
    v = ofsts(:,j)*ones(1,L) + ...
      ones(K,1)*nghbr_ofsts(:,j)';
    n_ofsts(:,j) = v(:);
  end
  
  [needed_ofsts, ~, ofsts_list] = unique(n_ofsts, 'rows');
  needed_ofsts = SimpleFractions(needed_ofsts,denom);

  nghbr_list = reshape(ofsts_list(K+1:end), K, (L-1));
  ofsts_list = ofsts_list(1:K);
end

