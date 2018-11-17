function psnr = vidPSNR( ref, tst, ref_skip, tst_skip, nfrms, rep_freq)
  %vidPSNR compute PSNR beteween YUV video files
  %  ref: reference file name (should be a JSON file)
  %  tst: test file name (should be a YUV file)
  %  ref_skip: number of frames to skip in refrence file
  %  tst_skip: number of frames to skip in test file
  %  nfrms: Specifies the number of frames to process.  If positive, this
  %         is the number of frames. Otherwise, the negative value is added
  %         to the number of frames available in the test file. Therefore,
  %         0 means use all frames in the test file (excluding the skipped
  %         ones)
  %  rep_freq: If positive, PSNR is written out every rep_freq frame
  ref_info = read_raw_video(ref, 0, ref_skip+1);
  tst_info = ref_info.copy();
  tst_info.handle = -1;
  tst_info.path = tst;
  tst_info = read_raw_video(tst_info, 0, tst_skip+1);
 
  if rep_freq > 0
    step = rep_freq;
    mode = 1;
  else
    step = 1;
    mode = 0;
  end
  
  vid_cmpr = VidCompare(ref_info.getPixelMax(), mode, ref_info, 0);
  
  if nfrms <= 0
    nfrms = tst_info.n_frames - tst_skip + nfrms;
  end
  for k=1:step:nfrms
    nf = min(step, nfrms-k+1);
    [tst_info,tst_data,err_msg] = read_raw_video(tst_info, nf);
    if ~isempty(err_msg)
      error('Frame %d: %s', k, err_msg);
    end
    vid_cmpr.update(tst_data);
  end 
  
  [psnr,~]= vid_cmpr.getPSNR();
end

