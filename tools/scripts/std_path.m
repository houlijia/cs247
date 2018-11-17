[mpath,~,~] = fileparts(mfilename('fullpath'));
path(mpath,path);

set_mex_path('std');


% This script sets the path, but eliminates all the 'dbg' directories, thus mex
% functions are always selected from the non-debug version in the 'bld' directories.

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
% sdbg = [filesep, 'dbg'];
% sdbgs = [sdbg, filesep];
% find_dbg = @(x) ~isempty(strfind(x, sdbgs)) || ...
%   (length(x) >= length(sdbg) && strcmp(sdbg, x(end-length(sdbg)+1:end)));
% c = strsplit(genpath(trunk), pathsep);
% c(cellfun(find_dbg, c)) = [];
% restoredefaultpath
% path(strjoin(c, pathsep) ,path(mlseq,path));

