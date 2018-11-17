function  sml_io = brief_tcp_dec( output_id, dec_id, spec )
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
    
    if nargin < 2
      dec_id = 'tcp_*';
      
      if nargin < 1
        output_id = 'd_*';
      end
    end
  end
  output_id = regexprep(output_id, '*',...
          datestr(now,'yyyymmdd_HHMM'),'once');
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
  dec_opts = struct('disp',0);
  end

  if isfield(spec, 'proc_opts')
    proc_opts = spec.proc_opts;
  else
    proc_opts = struct();
  end
  proc_opts.output_id = output_id;
  proc_opts.dec_id = dec_id;
%   files_def = '<foreman_news_tcp_io.json';
  files_def = brief_tcp_files_def('135.112.178.79:9000');

  sml_io =  CSVidDecoder.doSimulation(anls_opts,dec_opts,files_def, proc_opts);
  
end

