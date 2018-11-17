function pets_test_anls(output_id)
  pets_test_start = tic;
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
  
  anls_opts = struct('fxd_trgt',true,'nrm_exp',1,...
    'm_range','&[6,6;2,2]', 'm_step_numer','&[1,1;1,1]', ...
    'm_step_denom','&[1;2]',...
    'chk_bgrnd', struct('mx_avg',0, 'mn_dcd',2, 'thrsh', 2)...
    );
  % anls_opts = [];
  
  dec_opts = [];
  % dec_opts = struct('Q_msrmnts',0,'max_int_iters',4);
  % dec_opts = struct();
  proc_opts = struct('output_id',output_id,...
    'case_id', 'Mr<Mr><Ma>Q<Qm><Lc><Lg>', 'prefix', '] ');
  files_def='<PETS2000_test_enc.json';
  CSVidDecoder.doSimulation(anls_opts, dec_opts, files_def,...
    proc_opts)
  toc(pets_test_start);
end
