set_mex_path('dbg');


% This script sets the path, but eliminates all the 'bld' directories, thus mex
% functions are always selected from the debug version in the 'dbg' directories.

% [mpath,~,~] = fileparts(mfilename('fullpath'));
% cd(mpath)
% cd ..
% trunk = pwd;
% fsep = filesep();
% if strcmp(fsep,'\')
%     m_str = '\\trunk';
% else
%     m_str = sprintf('%strunk$', filesep());
% end
% if isempty(regexp(trunk, m_str, 'ONCE'))
%     cd ..
% end
% cd ..;
% cd mlseq
% mlseq = pwd;
% cd ..
% cd test_data
% sbld = [filesep, 'bld'];
% sblds = [sbld, filesep];
% find_bld = @(x) ~isempty(strfind(x, sblds)) || ...
%   (length(x) >= length(sbld) && strcmp(sbld, x(end-length(sbld)+1:end)));
% c = strsplit(genpath(trunk), pathsep);
% c(cellfun(find_bld, c)) = [];
% restoredefaultpath
% path(strjoin(c, pathsep) ,path(mlseq,path));

