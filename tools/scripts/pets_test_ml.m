pets_test_start = tic;
[mpath,~,~] = fileparts(mfilename('fullpath'));
cd(mpath)
cd ..
trunk = pwd;
m_str = sprintf('%strunk$', filesep());
if isempty(regexp(trunk, m_str, 'ONCE'))
    cd ..
end
cd ..
cd test_data
enc_opts = '<corr_pets_tst_ml.json';
anls_opts = struct('fxd_trgt',true,'nrm_exp',1);
dec_opts = [];
proc_opts = struct('case_id', 'Mr<Mr><Ma>Q<Qm><Lc><Lg>', 'prefix', '] ');
files_def='<PETS2000_test_enc.json';
CSVidCodec.doSimulationCase(enc_opts, anls_opts, dec_opts, files_def,...
    proc_opts)
toc(pets_test_start);
