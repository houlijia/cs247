function mrk_blk = mark_blk_boundaries( blk, ovrlp, ~, black, white)
    %mark_blk_boundaries - mark frame and search boundaries in a block
    %Input:
    %  blk - a cell array containing 3 dimensional blocks.  Only the first
    %        one (Y) is modified.
    %  ovrlp - frame overlap (h,v,t) 
    %  conv_rng - correlation range - marked in white.  presently disabled
    %  black - value of the color black
    %  white - value of the color white
    
    if iscell(blk)
        mrk_blk = blk;
        blk = mrk_blk{1};
        cell_inp = true;
    else
        cell_inp = false;
    end
    
    ovrlp = floor(ovrlp(1,:)/2);
    
    % Mark inner boundaries of frame (black)
    blk = pix_blk_draw(blk, black, [1+ovrlp(1),-ovrlp(1)], ...
        [1:ovrlp(2), -ovrlp(2)+1:0], [], [0,1,1]);
    blk = pix_blk_draw(blk, black, [1:ovrlp(1), -ovrlp(1)+1:0], ...
        [1+ovrlp(2),-ovrlp(2)], [], [1,0,1]);
    
    % Highlight first frame One (black, one pixel inside
    % boundary, as well as overlay)
    blk = pix_blk_draw(blk, white, [2+ovrlp(1),-1-ovrlp(1)], ...
        [1:ovrlp(2)+1, -ovrlp(2):0], 1, [0,1,0]);
    blk = pix_blk_draw(blk, white, [1:ovrlp(1)+1, -ovrlp(1):0], ...
        [2+ovrlp(2),-1-ovrlp(2)], 1, [1,0,0]);

%     % Mark conv_rng area in white
%     if any(conv_rng(1:2))
%         blk = pix_blk_draw(blk, white,...
%             [1+conv_rng(1), -conv_rng(1)], [1:conv_rng(2), -conv_rng(2)+1:0],...
%             [], [0,1,1]);
%         blk = pix_blk_draw(blk, white,...
%             [1:conv_rng(1), -conv_rng(1)+1:0], [1+conv_rng(2), -conv_rng(2)],...
%             [], [1,0,1]);
% 
%         % Highlight first frame of conv_rng
%         blk = pix_blk_draw(blk, white,...
%             [2+conv_rng(1,1), -conv_rng(1,1)-1], [1:conv_rng(1,2)+1, -conv_rng(1,2):0],...
%             1, [0,1,0]);
%         blk = pix_blk_draw(blk, white,...
%             [1:conv_rng(1,1)+1, -conv_rng(1,1):0], [2+conv_rng(1,2), -conv_rng(1,2)-1],...
%             1, [1,0,0]);
%     
%     end

    if cell_inp
        mrk_blk{1} = blk;
    else
        mrk_blk = blk;
    end
end

