[mpath,~,~] = fileparts(mfilename('fullpath'));
cd(mpath)
cd ..
cd ..
cd test_data
files_def='<PETS2000_test_enc.json';
proc_opts=struct('output_id','pets_20130701_1811','case_id','1');
simul_io_data = CSVidDecoder.doSimulation(struct(),struct(),...
    files_def, proc_opts);
