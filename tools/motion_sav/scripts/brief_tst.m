[mpath,~,~] = fileparts(mfilename('fullpath'));
cd(mpath)
cd ..
trunk = pwd;
fsep = filesep();
if strcmp(fsep,'\')
    m_str = '\\trunk$';
else
    m_str = sprintf('%strunk$', filesep());
end
if isempty(regexp(trunk, m_str, 'ONCE'))
    cd ..
end
cd ..
cd test_data
enc_opts = struct('msrmnt_mtrx',struct('type','SensingMatrixWH', 'args',struct()),...
    'blk_size','&[72 88 2]',...
    'n_frames',2,...
    'msrmnt_input_ratio', {{0.25,0.15}}, 'random',struct('seed',1000),...
    'lossless_coder', {{'&CSVideoCodecInputOutputData.LLC_INT', '&CSVideoCodecInputOutputData.LLC_AC'}});
% enc_opts = struct(...
%     'msrmnt_mtrx',struct('type','SensingMatrixWH',...
%         'args',struct('rnd_type',1)),...
%     'blk_size','&[72 88 2]',...
%     'n_frames',2,...
%     'msrmnt_input_ratio', {0.25,0.15}, 'random',struct('seed',1000));
anls_opts = [];
dec_opts = struct('disp',0);
ids = struct('output','*','case','Mr<Mr>Q<Qm>,<Qa><Lc><Lg>');
files_def = '<foreman_news_io.json';
proc_opts = struct();
sml_io =  CSVidCodec.doSimulation(enc_opts,[],dec_opts(),files_def, proc_opts);
