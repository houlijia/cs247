function vid_diff(inp, out, opts)
  
  if nargin < 3
    opts = CS_EncParams;
  elseif ~isa(opts, 'CS_EncParams')
    opts = CS_EncParams(opts);
  end
  
  rng(opts.random.seed, 'twister');
  
  raw_vid_in = VidBlocksIn(opts.blk_size, struct(...
    'ovrlp', opts.blk_ovrlp,...
    'monochrom', ~opts.process_color,...
    'w_type', opts.wnd_type),...
    inp, opts.n_frames, opts.start_frame-1);
%   enc_info.raw_vid = raw_vid_in.vid_info;
%   enc_info.raw_size = raw_vid_in.vid_size;
  if opts.n_frames  > raw_vid_in.vid_size(1,3)
    opts.setParams(struct('n_frames', raw_vid_in.vid_size(1,3)));
  end
  pxmx = raw_vid_in.vid_info.getPixelMax()+1;
  
  vid_out = VidBlocksOut(out, false, raw_vid_in, VidBlocker.BLK_STT_WNDW);
  
  zero_ext = [opts.zero_ext_b; opts.zero_ext_f];
  
  nxt_blk_indx = [1,1,1];
  while ~isempty(nxt_blk_indx)
    cur_blk_indx = nxt_blk_indx;
    vid_region = VidRegion(cur_blk_indx, raw_vid_in, zero_ext); 
    
    [inp_blk, nxt_blk_indx] = raw_vid_in.getBlks(cur_blk_indx);
    
    out_blk = vid_region.multiDiffExtnd(inp_blk, opts.blk_pre_diff);
    for clr = 1:length(out_blk)
      ob = out_blk{clr};
      sz = size(ob);
      ob = floor(min(abs(ob(:)),pxmx));
      out_blk{clr} = reshape(ob,sz);
    end
    
    vid_out = vid_region.putIntoBlkArray(out_blk, vid_out);
    vid_out.writeReadyFrames(raw_vid_in.vid_info);
    fprintf('blk %s done\n', show_str(cur_blk_indx));
  end
end

