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
enc_opts = '<corr_pets_tst_lclDFT.json';
anls_opts = struct('fxd_trgt',true,'nrm_exp',1,...
    'm_range','&[6,6;2,2]', 'm_step_numer','&[1,1;1,1]', ...
    'm_step_denom','&[1;2]');
dec_opts = [];
proc_opts = struct('case_id', 'Mr<Mr><Ma>Q<Qm><Lc><Lg>', 'prefix', '] ');
files_def='<PETS2000_test_enc.json';
CSVidCodec.doSimulationCase(enc_opts, anls_opts, dec_opts, files_def,...
    proc_opts)
toc(pets_test_start);
