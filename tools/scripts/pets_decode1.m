tic
[mpath,~,~] = fileparts(mfilename('fullpath'));
cd(mpath)
cd ..
cd ..
cd test_data
files_def='<PETS2000_test_enc1.json';
anls_opts= struct();
dec_opts = struct('max_out_iters',0, 'max_int_iters',0,'max_iters',0);
simul_io_data = NewCSVideoDecoder.doSimulation(anls_opts,dec_opts,'pets_20130701_1811',...
    '1',files_def);
toc
