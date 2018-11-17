function result = CSVidDec(anls_opts, dec_opts, proc_opts, ...
    enc_vid, output, fldef)
  %CSVidProcess - Decode a single CS Vid file
  %  Input:
  %    anls_opts - Analysis options ([] = no analysis)
  %    dec_opts - decoding options ([] = no decoding)
  %    proc_opts - processing options ([] = struct())
  %    enc_vid - input file (csvid)
  %    output - decoded output file ([] = no decode)
  %    fldef - (optional) a struct containing other files, as specified
  %            by FileDef
  
  if nargin < 6
    fldef = struct();
  end
  fldef.enc_vid = enc_vid;
  if ~isempty(output)
    fldef.output = output;
  end
  
  if isempty(proc_opts)
    proc_opts = struct();
  end
  if ~isfield(proc_opts, 'prefix')
    proc_opts.prefix = ']';
  end
  prefix = proc_opts.prefix;
  
  if ~isempty(anls_opts)
    if ~isa(anls_opts,'CS_AnlsParams')
      anls_opts = CS_AnlsParams(anls_opts);
    end
    fprintf('%s\n',anls_opts.describeParams(prefix));
  else
    fprintf('No analysis\n');
  end
  
  if ~isempty(dec_opts)
    if ~isa(dec_opts, 'CS_DecParams')
      dec_opts = CS_DecParams(dec_opts);
    end
    fprintf('%s\n',dec_opts.describeParams(prefix));
  else
    fprintf('No decoding');
  end
  
  if isequal(dec_opts.init, -1) && isfield(fldef, 'input');
    dec_opts.ref = fldef.input;
  end
  
  dec = CSVidDecoder(fldef, anls_opts, dec_opts);
  dec.setPrefix(prefix);
  if isfield(proc_opts, 'blk_rng')
    dec.setBlkRange(proc_opts.blk_rng);
  end
  
  result = dec.run(enc_vid, proc_opts);
end

