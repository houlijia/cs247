function [svn_ver, err_no] = getSvnVersion( )
  % Returns a Subversion version string or a numeric error code and error
  % message.
  
%   persistent svn_version;
%   
%   if ~isempty(svn_version)
%     svn_ver = svn_version;
%     return
%   end

  persistent svnversion_prog;
  cur_dir = pwd;
  [mpath,~,~] = fileparts(mfilename('fullpath'));
  cd(mpath)
  cd ..
 
  if ~isempty(svnversion_prog)
    [err_no, svn_ver] = system(svnversion_prog);
  else
    err_no = 1;
  end
  
  if err_no
    [err_no, svn_ver] = system('svnversion');
    if ~err_no
      svnversion_prog = 'svnversion';
    elseif ispc()
      svnvers = {'%SystemDrive%\cygwin\bin\svnversion',...
        '%SystemDrive%\cygwin64\bin\svnversion'};
      for k=1:length(svnvers)
        [err_no, svn_ver] = system(svnvers{k});
        if ~err_no
          svnversion_prog = svnvers{k};
          break
        end
      end
    end
  end
  
  if ~err_no
    svn_ver = regexprep(svn_ver, '\n+$','');
  end
  
  cd(cur_dir);
  
%   svn_version = svn_ver;
end

