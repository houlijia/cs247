[mpath,~,~] = fileparts(mfilename('fullpath'));
cd(mpath)
cd ..
cd ..
cd test_data
enc_opts = '<pets_enc_tst.json';
anls_opts = [];
dec_opts = [];
proc_opts = struct('case_id', 'Mr<Mr><Ma>Q<Qm>', 'prefix', ']');
files_def='<PETS2000_test_enc.json';
CSVidCodec.doSimulationCase(enc_opts, anls_opts, dec_opts, files_def,...
    proc_opts)
