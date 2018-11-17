function [ UVblk_size, UVovrlp ] = calc_UVblk_size(Ysize, UVsize, Yblk_size, Yovrlp )
    %calc_UVblk_size Calculate the block size and overlap of the U,V components
    %given the block size and overlap of the Y components.
    %
    %All input and output arguments are 1x3 arrays of [height, width,
    %frames].
    %
    % INPUT arguments:
    % Ysize - size of Y input
    % UVsize - size of UV input
    % Yblk_size - block size of the Y components
    % Yovrlp - overlap of blocks in the Y component.
    %
    % OUTPUT arguments:
    % UVblk_size - size of UV blocks
    % UVovrlp - overlap of UV blocks
    
    % Check consistency of input
    % - Check that all inputs are of size (1,3)
    if any([size(Yblk_size), size(Yovrlp), size(Ysize), size(UVsize)] ~= ...
            [1 3 1 3 1 3 1 3])
        err = MException('InputChk:OutOfRange', 'Some input arrays are not of size (1,3)');
        throw(err)
    end
    % - Check that all inputs are positive integers
    all_inp=[Yblk_size, Yovrlp, Ysize, UVsize];
    if any([mod(all_inp,1), ([Yblk_size, Ysize, UVsize] <= 0), (Yovrlp < 0)])
        err = MException('InputChk:OutOfRange', 'input arrays contain non-positive integer');
        throw(err)
    end
    % - Check that overlap is not too big
    if any(2*Yovrlp > Yblk_size)
        err = MException('InputChk:OutOfRange', 'overlap too large');
        throw(err)
    end
    % - Compute and check UVratio
    UVratio = round(Ysize ./ UVsize);
    if any(UVratio <=0)
        err = MException('InputChk:OutOfRange', 'Illegal ratio between Y and UV sizes');
        throw(err)
    end
    if any([mod(Yblk_size,UVratio), mod(Yovrlp,UVratio)])
        err = MException('InputChk:OutOfRange', ...
            'ratio between Y and UV sizes does not divide Y block size of Y overlap');
        throw(err)
    end
    
    UVblk_size = Yblk_size ./ UVratio;
    UVovrlp = Yovrlp ./ UVratio;
end
