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
anls_opts = [];
dec_opts = struct();
files_def = '<all_io.json';
proc_opts = struct('output_id', '20140729_1659',...
  'dec_id', 'no_loss');
sml_io =  CSVidDecoder.doSimulation(anls_opts,dec_opts(),files_def, proc_opts);
