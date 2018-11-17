function blk_cnt = calc_blk_cnt( Ysize, Yblk_size, Yovrlp )
    %calc_blk_cnt returns the number of blocks in each dimension.
    %
    %All input and output arguments are 1x3 arrays of [height, width,
    %frames].
    %
    % INPUT arguments:
    % Ysize - size of Y input
    % Yblk_size - block size of the Y components
    % Yovrlp - overlap of blocks in the Y component.
    %
    % OUTPUT arguments:
    % blk_cnt is a 1x3 array of number of blocks in each dimension
    
    % Check consistency of input
    % - Check that all inputs are of size (1,3)
    if any([size(Yblk_size), size(Yovrlp), size(Ysize)] ~= [1 3 1 3 1 3])
        err = MException('InputChk:OutOfRange', 'Some input arrays are not of size (1,3)');
        throw(err)
    end
    % - Check that all inputs are positive integers
    all_inp=[Yblk_size, Yovrlp, Ysize];
    if any([mod(all_inp,1), (Yblk_size<=0), (Ysize<=0)])
        err = MException('InputChk:OutOfRange', 'input arrays contain non-positive integer');
        throw(err)
    end
    % - Check that overlap is not too big
    if any(2*Yovrlp > Yblk_size)
        err = MException('InputChk:OutOfRange', 'overlap too large');
        throw(err)
    end
    
    blk_cnt = ones(1,3) + ceil(max(0,(Ysize-Yblk_size))./(Yblk_size-Yovrlp));
end
