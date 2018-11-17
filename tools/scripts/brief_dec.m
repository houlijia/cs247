function  sml_io = brief_dec( output_id, dec_id, spec )
  %brief_dec - decode output of brief_tst
  
  [mpath,~,~] = fileparts(mfilename('fullpath'));
  cd(mpath)
  cd ..
  trunk = pwd;
  if strcmp(filesep(),'\')
    m_str = '\\trunk';
  else
    m_str = sprintf('%strunk$', filesep());
  end
  if isempty(regexp(trunk, m_str, 'ONCE'))
    cd ..
  end
  cd ..
  cd test_data
  if nargin < 3
    spec = struct();
  end
  if isfield(spec,'anls_opts')
    anls_opts = spec.anls_opts;
  else
    anls_opts = [];
  end
  
  if isfield(spec,'dec_opts')
    dec_opts = spec.dec_opts;
  else
    % dec_opts = struct('disp',0, 'expand_level', VidBlocker.BLK_STT_INTRPLT);
    % dec_opts = struct('disp',1, 'expand_level', VidBlocker.BLK_STT_RAW, ...
    %   'beta_A0', 0.01, 'beta_D0', 0.01, 'max_int_iters', 10, 'sparsifier', ...
    %   struct('args', struct('b_stt', VidBlocker.BLK_STT_INTRPLT)));
    % dec_opts = struct('disp',1, 'expand_level', VidBlocker.BLK_STT_RAW, ...
    %   'beta_A0', 0.01, 'beta_D0', 0.01, 'max_int_iters', 4, 'max_iters',20, 'sparsifier', ...
    %   struct('args', struct('b_stt', VidBlocker.BLK_STT_INTRPLT)));
    dec_opts = struct('disp',0, 'expand_level', 4, 'sparsifier', ...
      struct('args', struct('b_stt',4)), 'whole_frames', true);
%     dec_opts = struct('disp',0);
  end

  if isfield(spec, 'proc_opts')
    proc_opts = spec.proc_opts;
  else
    proc_opts = struct();
  end
  proc_opts.output_id = output_id;
  proc_opts.dec_id = dec_id;
  files_def = '<foreman_news_io.json';

  sml_io =  CSVidDecoder.doSimulation(anls_opts,dec_opts,files_def, proc_opts);
  
end

