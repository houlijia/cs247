function vid_sqr(inp, rnd_out, sqr_out, opts)
  
  if nargin < 4
    opts = CS_EncVidParams;
  elseif ~isa(opts, 'CS_EncVidParams')
    opts = CS_EncVidParams(opts);
  end
  
  if opts.sav_levels
    intrplt = pow2(opts.sav_levels);
  else
    intrplt = 1;
  end

  raw_vid_in = VidBlocksIn([], inp, VidBlocker.BLK_STT_WNDW, opts, intrplt);
  
  rng(opts.random.seed, 'twister');
  
%   enc_info.raw_vid = raw_vid_in.vid_info;
%   enc_info.raw_size = raw_vid_in.vid_size;
  if opts.n_frames  > raw_vid_in.vid_size(1,3)
    opts.setParams(struct('n_frames', raw_vid_in.vid_size(1,3)));
  end
  
  rnd_vid_out = VidBlocksOut(rnd_out, true, raw_vid_in, VidBlocker.BLK_STT_WNDW);
  sqr_vid_out = VidBlocksOut(sqr_out, false, raw_vid_in, VidBlocker.BLK_STT_WNDW);
  
  pxmx = raw_vid_in.vid_info.getPixelMax()+1;
  zero_ext = [opts.zero_ext_b; opts.zero_ext_f];
  
  nxt_blk_indx = [1,1,1];
  while ~isempty(nxt_blk_indx)
    cur_blk_indx = nxt_blk_indx;
    vid_region = VidRegion(cur_blk_indx, raw_vid_in, zero_ext); 
    
    [inp_blk, nxt_blk_indx] = raw_vid_in.getBlks(cur_blk_indx);
    inp_vec = vid_region.vectorize(inp_blk);
    
    rnd_vec = inp_vec .* (-1) .^ randi([0,1],size(inp_vec));
    rnd_vid_out = vid_region.putIntoBlkArray(rnd_vec, rnd_vid_out);
    rnd_vid_out.writeReadyFrames();
    
    sqr_vec = (rnd_vec .^ 2)/pxmx;
    sqr_vid_out = vid_region.putIntoBlkArray(sqr_vec, sqr_vid_out);
    nfr = sqr_vid_out.writeReadyFrames();
    if nfr
      fprintf('Finished temporal block %d\n', cur_blk_indx(3));
    end
  end
end

