function p = get_mex_path()
  p = regexp(path(),['mex.*exe' pathsep()], 'match');
end