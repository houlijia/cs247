function ok = chk_analyze_cnstrnt(enc_opt, cnstrnt)
  %chk_analyze_cnstrnt Check if enc_opt matches the constraing cnstrnt
  
  % enc_opt and cnstrnt are structs. The function returns OK if for each
  %         in cnstrnt there is a corresponding field in enc_opt with a
  %         matching value.
  
  ok = true;
  flds = fieldnames(cnstrnt);
  for f = 1:length(flds)
    fld = flds{f};
    
    if ~isfield(enc_opt, fld)
      ok = false;
      return
    elseif isstruct(cnstrnt.(fld))
      if ~isstruct(enc_opt.(fld)) || ~ chk_analyze_cnstrnt(enc_opt.(fld), cnstrnt.(fld))
        ok = false;
        return
      end
    elseif ~isequal(enc_opt.(fld), cnstrnt.(fld))
      ok = false;
      return
    end
  end
end
