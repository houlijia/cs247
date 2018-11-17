function write_str_to_file(dr, fname, str)
  % Saves a string in a file
  %    dr - directory
  %    fname - file name
  %    str - string name
  fid = fopen(fullfile(dr,fname),'wt');
  fprintf(fid, '%s\n', str);
  fclose(fid);
end

