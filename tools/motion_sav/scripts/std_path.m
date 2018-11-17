[mpath,~,~] = fileparts(mfilename('fullpath'));
cd(mpath)
cd ..
trunk = pwd;
fsep = filesep();
if strcmp(fsep,'\')
    m_str = '\\trunk';
else
    m_str = sprintf('%strunk$', filesep());
end
if isempty(regexp(trunk, m_str, 'ONCE'))
    cd ..
end
cd ..;
cd mlseq
mlseq = pwd;
cd ..
cd test_data
restoredefaultpath
path(genpath(trunk),path(mlseq,path));

