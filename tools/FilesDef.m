classdef FilesDef < handle
  % FilesDef is a class objects which can generate sets of I/O specs for
  % use in an experiment. 
  %
  % The term "I/O spec" usually refers to a disk file,but it can also be a
  % specification of a socket. Each set of I/O specs contains several I/O 
  % typespecs of different types (input, output, etc.). The allowed types 
  % are listed in FilesDef.std_types.
  %
  % If the I/O spec is a file path it is generated as the concatenation of 
  %   [<place> <base file name> <extension>]
  % <place> is the directory in which the file is found or placed and 
  % <extension> is specific to the type. 
  %
  % The possible output directories have the form:
  % The encoder directory, where encoding output files are stored is
  %
  %   output_dir[output_id]/case_dir[case_id]
  %
  % The decoder directory, where decoder output files are stored is
  %
  %   output_dir[output_id]/dec_dir[dec_id]/case_dir[case_id]
  %
  % if dec_dir is not specificed (usually when encoding is also done and
  % decoder output directory is the same as encoder output directory) the
  % decoder directory is the same as the encoder directory.
  
  properties (Constant, Access = private)
    % Symbolic names
    I=1; % Input
    E=2; % Encoder output
    D=3; % Decoder output
  end
  
  properties (Constant)
    % A struct array containing the common data typse and their standard
    % extensions. The field 'meaning' are mereley for documentation:
    %  'type' is the file type name
    %  'ext' is the default extension for this file type
    %  'place' specifies where the file is expected to be
    %      placed:
    %         FilesDef.I - input,
    %         FilesDef.E - encoder output,
    %         FilesDef.D - decoder output,
    %  'meaning' - Explanation for documentation purposes
    
    std_types =cell2struct({...
      'input', '.json', FilesDef.I, 'Input: Raw video';...
      'inp_copy', '.i.yuv', FilesDef.E, 'Copy of the input';...
      'enc_vid', '.csvid', FilesDef.E, 'Encoder output, with CS data stream';...
      'output', 'o.yuv', FilesDef.D, 'Reconstructed video';...
      'txt', '.txt', FilesDef.D, 'Text summary of execution';...
      'mat', '.mat', FilesDef.D, 'Matlab state at end of execution';...
      'inp_anls', '.im.csv', FilesDef.E, 'Analysis results of input (CSV)';...
      'inp_sav', '.im.sav.txt', FilesDef.E, 'SAV of input (SAV)';...
      'inp_mark', '.im.yuv', FilesDef.E, 'Raw video + motion marks based on input';...
      'inp_vanls', '.imv.csv', FilesDef.E, ...
        'Analysis results of input using vector shifts (CSV)';...
      'inp_vsav', '.imv.sav.txt', FilesDef.E, 'SAV of input using vector shifts (SAV)';...
      'inp_vmark', '.imv.yuv', FilesDef.E, 'Raw video + motion marks using vector shifts';...
      'enc_anls', '.em.csv', FilesDef.E, 'Analysis results of raw measurements (CSV)';...
      'enc_sav', '.em.sav.txt', FilesDef.E, 'Analysis results of raw measurements (SAV)';...
      'enc_mark', '.em.yuv', FilesDef.E, 'Raw video + motion marks based on raw measuremnts';...
      'dec_anls', '.dm.csv', FilesDef.D, 'Analysis results of decoded measuremens (CSV)';...
      'dec_sav', '.dm.sav.txt', FilesDef.D, 'Analysis results of decoded measuremens (SAV)';...
      'ref_mark', '.rm.yuv', FilesDef.D, ...
      'Raw video + motion marks based on decoded measuremnts';...
      'dec_mark', '.dm.yuv', FilesDef.D, ...
        'Reconstructed video + motion marks based on decoded measuremnts';...
      'dec_slct_mark', '.dm_sl.yuv', FilesDef.D, ...
        'Selectively reconstructed video + motion marks based on decoded measuremnts';...
      'enc_pre_diff','.edff.yuv', FilesDef.E, 'Input video after performing pre_diff';...
      'dec_pre_diff','.ddff.yuv', FilesDef.D, 'Decoded video before reversing pre_diff';...
      'tst_pre_diff', '.dffo.yuv', FilesDef.E, ...
        'Test file generated by reverse reverse pre_diff on the enc_pre_diff data';...
      'err_pre_diff', '.dffe.yuv', FilesDef.D, ...
        'Difference between dec_pre_diff and enc_pre_diff';...
      'dec_ref_diff', '.dffr.yuv', FilesDef.D, ...
        'copy of the reference pre_diff fed to the decoder ';...
      'cnv_enc', '.cnv.csvid', FilesDef.D,...
        'Compressed video file with converted compression';...
      'cnv_mat', '.cnv.mat', FilesDef.D, 'Matlab state after conversion'
      },...
      {'type', 'ext', 'place', 'meaning'},...
      2);
  end
  
  properties (SetAccess=private)
    sock = [];  % If not empty, a CodeSourceTCP or CodeDstTCP, depending on
                % params.do_enc
  end
  
  properties (Access=private)
    files;           % Base file names
    
    % The following structs have keys as field names
    types=struct(); % values are extensions.
    dirs_indx = struct(); % values are std_types.place values
    dirs = [];
    
    input_dir='';
    case_dir_pattern='';
    % These two structs have entries for output_dir, case_dir, dec_dir
    smbl = struct();  % Symbolic values (with place holders)
    actl = struct();  % Actual values (resolved)
    
  end
  
  methods
    % FilesDef - Constructor,  builds
    % files[] from spec.
    %
    % Input:
    % spec -  Specifies I/O paths, extensions, files and directories. Can
    %   be one of the following
    %     - A struct with the fields:
    %        * 'types' - Defines the I/O specs. Can one of. 
    %          - a struct, where each field is a name of a type and
    %            the field value is the definition of the type. 
    %          - a cell array, each cell containing one of:
    %            ~ A character string specifying the name of a type. The
    %              definition of the type is assumed to be empty.
    %            ~ a struct, where each field is a name of a type and
    %              the field value is the definition of the type.
    %          In the above if any struct has a field with a field name of 
    %          'ignore' and a true value, the struct is ignored.
    %          The definition of types can be one of the following:
    %          - empty, in which case the definition is taken from
    %            FilesDef.std_types
    %          - A string which is interpreted as the extension.
    %          - A struct which can define either a file extension or a TCP
    %            socket interface and is intended to be used by 
    %            CodeDest.constructCodeDest() or 
    %            CodeSource.constructCodeSource(). 
    %            For a file there is a single field:
    %            ~ ext - the file extension. 
    %            For a TCP Socket the following fields are used
    %            ~ addr - Specification of IP addresses in the form:
    %                <host>:<port>[::<l_host>[:<l_port>]]
    %            ~ sndr_is_srvr - (logical) If true, the encoder is opened
    %              in server mode (listen) and the decoder is opened in
    %              client mode (connect). If false, the roles are reversed.
    %              Default: false.
    %            ~ cnct_timeout - Maximal waiting time (seconds) for
    %              connection (applies only to the server). 0=indefinite, 
    %            ~ recv_timeout - Maximal waiting time (seconds) for
    %              receiving data (applies only to the decoder)
    %          files. Each struct element has one field (type name)
    %          with a string value (type extension).
    %        * 'names - an array of strings with files base names
    %        * 'input_dir' (optional) - directory of input
    %        * 'output_dir' (optional) - output_dir directory
    %        * 'dec_dir' (optional) - directory of decoded files,
    %              ignored if params.do_enc is true
    %        * 'case_dir (optional) - if present specifies a
    %          subdirectory which is specific to all files of a
    %          specific case.
    %     - A JSON string which resolves to the above struct
    %     - A string beginning with "<", inerpreted as the name of a
    %       file containing the JSON string.
    % params - (optional) a struct containing optional arguments, which
    %        can be:
    %    - fields: If present it is a cell array of strings
    %         specifying the keys which are needed. spec.types
    %         should contain all these keys. keys in spec.types which
    %         are not in fields are ignored (an error occurs if a
    %         string in params.fields is missing in spec.types).
    %    - do_enc: If present and true dec_dir is forced to be
    %         an empty string.
    %    - identifier - a struct suitable for calling specifyDirs(). If
    %         present files_def.specifyDirs(params.identifier) will be
    %         called.
    function files_def = FilesDef(spec, params)
      if nargin < 2
        params = struct();
      end
      
      % Parse spec
      info = ProcessingParams.read_opts(spec);
      if isstruct(info)
        info = ProcessingParams.eval_fields(info);
      end
      
      % Set directories
      files_def.input_dir = files_def.getDirNames(info, 'input_dir');
      files_def.case_dir_pattern = files_def.getDirNames(info,...
        'case_dir_pattern');
      for nm = {'output_dir', 'case_dir', 'dec_dir'}
        files_def.smbl.(nm{1}) = files_def.getDirNames(info, nm{1});
      end
      if isfield(params, 'do_enc') && params.do_enc
        files_def.smbl.dec_dir = '';
      end
      files_def.actl = files_def.smbl;
      
      % Special handling of case_dir_pattern
      if isempty(files_def.case_dir_pattern)
        files_def.case_dir_pattern = files_def.smbl.case_dir;
      end
      % Remove terminating file separator
      files_def.case_dir_pattern = ...
        files_def.case_dir_pattern(1:end-length(filesep()));
      
      if isfield(params, 'identifier')
        files_def.specifyDirs(params.identifier);
      end
      
      % check consistency of info.types
      [files_def.types, files_def.sock] = FilesDef.chkTypes(info.types, params);
      
      files_def.files = info.names;
      
    end  % FilesDef
    
    % specifyDirs replaces place holder in paths by a given
    % identififier. The path can be one of 'output', 'case' or 'dec'.
    % Input:
    %   indentifier - a struct with any of the following fields:
    %       'output', 'case', 'dec'. The field values are the
    %       string identifiers to replace the place holder in the respective
    %       paths. If the field values are numeric/logical, a value of
    %       true causes no action and a value of 'false' delete the
    %       specified field (useful mainly for 'dec').
    %   place_holder (optional) - a struct with any of the following fields:
    %       'output', 'case', 'dec'. The field values are the place
    %       holder to be replaced. If the struct or the field is
    %       missing '*' is assumed.
    function specifyDirs(files_def, identifier, place_holder)
      flds = fieldnames(identifier);
      for ifld = 1:length(flds)
        fld = flds{ifld};
        path_name = [fld '_dir'];
        if ischar(identifier.(fld))
          if nargin > 2 && isfield(place_holder, fld)
            phldr = place_holder.(fld);
          else
            phldr = '*';
          end
          files_def.actl.(path_name) = regexprep(...
            files_def.smbl.(path_name),...
            phldr, identifier.(fld), 'once');
        elseif ~identifier.(fld)
          files_def.actl.(path_name) = '';
        end
      end
    end
    
    % set a path
    % Input:
    %   indentifier - a struct with any of the following fields:
    %       'output', 'case', 'dec'. The field values are the
    %       strings  to set as in the respective
    %       paths.
    function setDirs(files_def, identifier)
      sep = filesep();
      rg = [regexprep(sep, '\', '\\\\') '$'];
      flds = fieldnames(identifier);
      for ifld = 1:length(flds)
        fld = flds{ifld};
        path_name = [fld '_dir'];
        vl = identifier.(fld);
        if isempty(regexp(vl, rg, 'once'))
          val = [vl sep];
        else
          val = vl;
        end
        switch path_name
          case 'input_dir'
            files_def.input_dir = val;
          case {'output_dir', 'case_dir', 'dec_dir'}
            files_def.actl.(path_name) = val;
        end
      end
    end
    
    % Initialize the process of getting files.
    % Input:
    %   files_def - this object
    % Output:
    %   indx - the index of the first object to return
    %   err_msg - if present returns error message if directory
    %         creation failed or empty string. If missing an error
    %         causes an exception.
    function [indx, err_msg] = init_getFiles(files_def)
      err_msg = files_def.makeDirs();
      if ~isempty(err_msg) && nargout < 2
        error('%s', err_msg);
      end
      
      indx = 1;
    end
    
    function istp = isType(files_def, type_name)
      istp = isfield(files_def, type_name);
    end
    
      
    % Computes and returns a struct defining the I/O files
    % corresponding to the base file corresponding to indx.
    %
    % Input:
    %   files_def: This object.
    %   indx: Index of base file to use.
    % Output:
    %   fdef: A struct in which the field names are
    %       are file types, values are file paths with appropriate
    %       extensions.  Note that all files get the path determined by
    %       output_dir/case_dir, except for the fdef.input which gets
    %       the path corresponding to input_dir. If index is beyond the
    %       number of base files, [] is returned.
    %   out_indx: optional. Returns indx incremented by 1
    %   name: The base name used to create the files
    function [fdef, out_indx, name] = getFiles(files_def, indx)
      if indx > length(files_def.files)
        fdef = [];
        out_indx = indx;
        name = '';
        return
      end
      
      name = files_def.files{indx};
      fdef = files_def.getFilesByName(name);
      out_indx = indx+1;
   end
    
    % Computes and returns a struct defining the I/O files
    % corresponding to a particular base file.
    %
    % Input:
    %   files_def: This object.
    %   name: base name.
    % Output:
    %   fdef: A struct in which the field names are
    %       are file types, values are file paths with appropriate
    %       extensions.  Note that all files get the path determined by
    %       output_dir/case_dir, except for the fdef.input which gets
    %       the path corresponding to input_dir. If index is beyond the
    %       number of base files, [] is returned.
    %   out_indx: optional. Returns indx incremented by 1
    %   name: The base name used to create the files
    function [fdef] = getFilesByName(files_def, name)
      fdef = files_def.types;
      flds = fieldnames(fdef);
      paths = cell(size(flds));
      l_paths = 0;
      for i=1:length(flds)
        type=flds{i};
        fdef.(type) = getTypeVal(fdef.(type));
      end

      function out = getTypeVal(inp)
        if iscell(inp)
          out = cell(size(inp));
          n_out = 0;
          for k=1:numel(inp)
            v = getTypeVal(inp{k});
            if ~isempty(v)
              n_out = n_out+1;
              if iscell(v)
                out(n_out) = v(:);
              else
                out{n_out} = {v};
              end
            end
          end
          out = out(1:n_out);
          out = vertcat(out{:});
        elseif isstruct(inp)
          if isfield(inp,'ext')
            out = [files_def.dirs{inp.dir_type} name inp.ext];
            prv = find(strcmp(paths(1:l_paths), out));
            if prv
              error('File %s is both %s and %s', out, ...
                flds{prv}, type);
            else
              l_paths = l_paths+1;
              paths{l_paths} = out;
            end
          else
            out = inp;
          end
        elseif isa(inp, 'CodeDest') || isa(inp, 'CodeSource')
          out = inp;
        else
          error('unexpected type of inp: %s', class(inp));
        end
      end
    end % getFilesByName()
    
    % Get a struct array for all files. Each entry in the array
    % contains the output of getFiles() for a particular index.
    % Input:
    %   files_def - this object
    %Outupt
    %   fdef_list - the struct array. If it is a character string it is
    %               an error message.
    %   err_msg - if present returns error message if directory
    %         creation failed or empty string. If missing an error
    %         causes an exception.
    function [fdef_list, err_msg] = getAllFiles(files_def)
      fld_names = fieldnames(files_def.types);
      fdef_list =  cell2struct(...
        cell(length(fld_names), length(files_def.files)),...
        fld_names, 1);
      
      [indx, err_msg] = files_def.init_getFiles();
      if ~isempty(err_msg)
        if nargout < 2
          error(err_msg);
        end
        return;
      end
      
      while true
        [fdef, out_indx] = files_def.getFiles(indx);
        if out_indx == indx
          break;
        end
        fdef_list(indx) = fdef;
        indx = out_indx;
      end
    end
    
    function dir_name = getEncoderDir(files_def)
      dir_name = [files_def.actl.output_dir files_def.actl.case_dir];
    end
    
    function dir_name = getDecoderDir(files_def)
      dir_name = [files_def.actl.output_dir files_def.actl.dec_dir...
        files_def.actl.case_dir];
    end
    
    function dir_name = getDecoderRoot(files_def)
      dir_name = [files_def.actl.output_dir files_def.actl.dec_dir];
      dir_name(end) = []; % remove directory separator
    end
    
    function dir_name = outputDir(files_def)
      dir_name = [files_def.actl.output_dir files_def.actl.dec_dir];
    end
    
    function err_msg = makeOutputDir(files_def)
      err_msg = files_def.makeDir(files_def.outputDir(), 'output');
      if ~isempty(err_msg) && nargout == 0;
        error('%s',err_msg);
      end
    end
    
    function dir_name = caseDir(files_def)
      dir_name = files_def.actl.case_dir;
    end
    
    function case_dirs = getCaseDirs(files_def)
      case_dirs = dir([files_def.actl.output_dir files_def.case_dir_pattern]);
      drs = cell(size(case_dirs));
      j=0;
      for k=1:length(case_dirs);
        if ~case_dirs(k).isdir || strcmp(case_dirs(k).name,'.') || ...
            strcmp(case_dirs(k).name,'..')
          continue;
        end
        j=j+1;
        drs{j} = case_dirs(k).name;
      end
      case_dirs = drs(1:j);
    end
    
    % makeDirs fills dirs with appropriate values. identifier and
    % place_holder are specified, specifyDirs is called before hand.
    % Input:
    %   files_def - this object
    % Output: reports success.
    %   err_msg - An error message. Empty if successful.
    function err_msg = makeDirs(files_def)
      files_def.dirs = {files_def.input_dir, ...
        files_def.getEncoderDir(), files_def.getDecoderDir()};
      
      err_msg = files_def.makeDir(files_def.dirs{2}, 'encoder');
      if ~isempty(err_msg)
        if nargout == 0
          error('failed making encoder dir %s\n  %s', ...
            files_def.dirs{2}, err_msg);
        end
        return;
      end
      err_msg = files_def.makeDir(files_def.dirs{3}, 'decoder');
      if ~isempty(err_msg)
        if nargout == 0
          error('failed making decoder dir %s\n  %s', ...
            files_def.dirs{3}, err_msg);
        end
        return;
      end
    end
    
  end   % Methods
  
  methods (Access=private)
    
  end   % Access = private
  
  methods (Access=private, Static=true)
    function [types, sock] = chkTypes(types, params)
      
      function types = getTypes(types)
        if ischar(types)
          types = struct(types,'');
        elseif isstruct(types)
          if isfield(types, 'ignore')
            if ~isempty(types.ignore) && types.ignore
              types = [];
            else
              types = rmfield(types,'ignore');
            end
          end
        elseif iscell(types)
          flds = cell(length(types),1);
          vals = cell(length(types),1);
          nt = 0;
          for j=1:length(types)
            v = getTypes(types{j});
            if isempty(v)
              continue;
            end
            nt = nt+1;
            flds{nt} = fieldnames(v);
            vals{nt} = struct2cell(v);
          end
          flds = vertcat(flds{1:nt});
          vals = vertcat(vals{1:nt});
          types = cell2struct(vals, flds, 1);
        else
          error('unexpected class of types: %s', class(types));
        end
      end
      
      std_type_names = struct2cell(FilesDef.std_types);
      std_type_names = std_type_names(1,:);
      
      types = getTypes(types);
      typenames = fieldnames(types);
      
      if isfield(params, 'fields')
        fields = params.fields;
        
        indx = ismember(fields, std_type_names);
        if ~all(indx)
          error('Field "%s" is an unrecognized type', ...
            fields{find(indx==0,1)});
        end
        
        indx = ismember(fields,typenames);
        if ~all(indx)
          error('Field "%s" is missing in types spec', ...
            fields{find(indx==0,1)});
        end
        
        indx = ismember(typenames, fields);
        if ~all(indx)
          indx = find(indx==1);
          typevals = struct2cell(types);
          typenames = typenames(indx);
          typevals = typevals(indx);
          types = cell2struct(typevals, typenames, 1);
        end
      end
      
      [indx, indpos] = ismember(typenames, std_type_names);
      if ~all(indx)
        error('Unrecognized type: %s', typenames{find(indx==0,1)});
      elseif length(unique(typenames)) < length(typenames)
        error('Repeated types');
      end


      function val = typeVal(val)
        if isempty(val)
          val = struct('ext', '');
        elseif ischar(val)
          val = struct('ext', val);
        end
        if isstruct(val) && isfield(val,'ext')
          if isempty(val.ext) 
            val.ext = FilesDef.std_types(indpos(k)).ext;
          end
          val.dir_type = FilesDef.std_types(indpos(k)).place;
        end
         
        if iscell(val)
          for j=1:length(val(:))
            val{j} = typeVal(val{j});
          end
        end
      end
      
      for k=1:length(typenames)
        typename = typenames{k};
        types.(typename) = typeVal(types.(typename));
      end
      
      % Set socket
      if isfield(types, 'enc_vid')
        [types.enc_vid, sock] = setSock(types.enc_vid);
      else
        sock = [];
      end
      
      function [val,sock] = setSock(val)
        sock = [];
        if iscell(val)
          for j=1:numel(val)
            [val{j},sock] = setSock(val{j});
            if ~isempty(sock)
              return;
            end
          end
        elseif isstruct(val) && isfield(val, 'addr')
          if isfield(params,'do_enc') && params.do_enc
            val = CodeDest.constructCodeDest(val);
            sock = val;
          else
            val = CodeSource.constructCodeSource(val);
            sock = val;
          end
          
        end
      end
    end
    
    function dir_name = getDirNames(info, field)
      if isfield(info,field);
        dir_name = info.(field);
        if ~strcmp(dir_name(end), '/')
          dir_name = [dir_name '/'];
        end
        
        % Change path separator to be platform specific
        sep=filesep();
        if ~strcmp(sep,'/')
          dir_name = regexprep(dir_name, '[/]', sep);
        end
      else
        dir_name = '';
      end
    end  % getDirNames
    
    function err_msg = makeDir(dir_path, dir_name)
      [status, emsg] = mkdir(dir_path);
      if ~status
        err_msg = sprintf('Failed creating %s directory (%s):\n\t%s', ...
          dir_name, dir_path, emsg);
      else
        err_msg = '';
      end
    end
    
  end % private static methods
  
  
end

