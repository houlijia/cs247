classdef ProcessingParams < matlab.mixin.Copyable
  %ProcessingParams Base class for classes defining processing
  %parameters.
  
  properties (Constant)
    % Default setting for parallel processing. 0=no parallel
    % processing. Otherwise, parallel processing with number of workers
    % = parallel * matlabpool(size)
    parallel = 0;
  end
  
  properties
    case_no = 0;
    n_cases = 0;
  end
  
  methods
    function obj=ProcessingParams(def)
      if nargin == 0
        return
      end
      if ~isa(def,'ProcessingParams')
        def = ProcessingParams.parse_opts(def);
      end
      obj.setParams(def);
    end
    
    % setParams sets the parameters of the object.
    % Input
    %   obj - this object
    %   params - can be:
    %         a struct whose field names are the same as some of the
    %            properties of obj.  The struct values are going to be set
    %            accordingly. If a field value is a string and it begins with
    %            an ampersand (&), the ampersand is removed and the field value
    %            is evalueated before being assigned.
    %         An object of a type which is a superclass of obj.  The
    %            non-constant fields in params are copied to obj.
    function setParams(obj, params)
      if isstruct(params)
        flds = fieldnames(params);
        for k=1:length(flds)
          fld = flds{k};
          obj.(fld) = params.(fld);
        end
      else
        mp = metaclass(params);
        obj_props = properties(obj);
        for k=1:length(mp.PropertyList)
          prop = mp.PropertyList(k);
          if ~any(strcmp(obj_props, prop.Name))
            continue;
          end
          if ~prop.Constant
            obj.(prop.Name) = params.(prop.Name);
          end
        end
      end
    end
    
    function s = getStruct(obj)
      s = struct();
      mp = metaclass(obj);
      for k=1:length(mp.PropertyList)
        prop = mp.PropertyList(k);
        if ~prop.Constant && strcmp(prop.GetAccess, 'public')
          s.(prop.Name) = obj.(prop.Name);
        end
      end
    end
    
    function s = show_str(obj, fmt, params)
      if nargin < 3
        params = struct();
        if nargin < 2
          fmt = '';
        end
      end
      s = show_str(obj.getStruct(),fmt,params);
    end
    
    function str = describeParams(obj, prefix)
      if nargin < 2
        prefix = '';
      end
      
      str = show_str(obj, '', struct('prefix', prefix, 'struct_marked', true));
    end
    
    function str = classDescription(obj)
      str = class(obj);
    end
    
    function jsn = getJSON(obj, file)
      s = obj.getStruct();
      if nargin == 1
        jsn = mat2json(s);
      else
        jsn = mat2json(s, file);
      end
    end
    
  end
  
  methods(Static)
    % setCaseNo receives a cell array of structs, and sets the fields
    % case_no and num_cases to the index of each element in the array
    % and to the number of elements in the  list, respectively.
    function cases = setCaseNo(cases)
      nc = numel(cases);
      for k=1:nc
        cases{k}.case_no = k;
        cases{k}.n_cases = nc;
      end
    end
    
    % getCases generates an array of structs, each of which can be used
    % to generate a ProcessingParams object.
    % Output:
    %   cases - a cell array - each cell contains a struct.
    % Input:
    %   spec - specification. Can be a character string, a struct or a
    %          cell array
    % Processing: First, if spec is a character string it is converted
    %   into a struct by read_opts(). Then getCases is applied
    %   recustively: If spec is a cell array, or a struct array of more
    %   than one element, getCases is applied to the contents of each
    %   cell,or to each entry in the struct array, respectively, and the
    %   outputs are concatenated. If spec is a single struct there are
    %   several options:
    %   * A field fld is found whose field name does not end with '__'
    %     and whose field value is a struct array, sa, with more than
    %     than one member, or a cell array, ca: A sequence of structs is
    %     created where the value of fld is sa(k), or ca{k}, respectively,
    %     where 1<=k<=length(ca) or 1<=k<=length(sa).  getCases is
    %     is applied to each member in the sequence and
    %     the resulting cell arrays are concatenated.
    %     each sa{n} and concatenate the results.
    %   * A field name fld ends with '_' (but nt with '__') and its value
    %     s is a struct array with one member. The original field fld is
    %     removed and the fields of s become fields of the original
    %     struct with the corresponding values.
    %   * Otherwise, the recursion ends.
    %     spec is parsed by eval_fields and then cases
    %     is a cell array of one entry containing spec.
    function cases = getCases(spec)
      spec = ProcessingParams.read_opts(spec);
      cases = cell(1,ProcessingParams.countCases(spec));
      n_done = 0;
      
      if iscell(spec)
        for k=1:numel(spec)
          case_list = ProcessingParams.getCases(spec{k});
          n_list = length(case_list);
          cases(n_done+1:n_done+n_list) = case_list;
          n_done = n_done + n_list;
        end
      elseif ~isstruct(spec)
        error('Unexpected type for spec');
      elseif numel(spec) ~= 1
        for k=1:numel(spec)
          case_list = ProcessingParams.getCases(spec(k));
          n_list = length(case_list);
          cases(n_done+1:n_done+n_list) = case_list;
          n_done = n_done + n_list;
        end
      else % must be a struct
        flds = fieldnames(spec);
        expanded = false;
        for k=1:length(flds)
          fld = flds{k};
          if regexp(fld,'__$','once')
            continue;
          end
          
          ca = spec.(fld);
          if iscell(ca) || (isstruct(ca) && numel(ca) ~= 1)
            for n=1:numel(ca)
              spc = spec;
              if iscell(ca)
                spc.(fld) = ca{n};
              else
                spc.(fld) = ca(n);
              end
              case_list = ProcessingParams.getCases(spc);
              n_list = length(case_list);
              cases(n_done+1:n_done+n_list) = case_list;
              n_done = n_done + n_list;
            end
          elseif regexp(fld,'_$','once')
            spc = rmfield(spec,fld);
            ca_flds = fieldnames(ca);
            for nf = 1:length(ca_flds);
              cfld = ca_flds{nf};
              spc.(cfld) = ca.(cfld);
            end
            case_list = ProcessingParams.getCases(spc);
            n_list = length(case_list);
            cases(n_done+1:n_done+n_list) = case_list;
          else
            continue;
          end
          expanded = true;
          break;
        end
        
        if ~expanded
          % Strip '__' from field names
          for k=1:length(flds)
            fld = flds{k};
            if regexp(fld,'__$','once')
              spec.(fld(1:end-2)) = spec.(fld);
              spec = rmfield(spec, fld);
            end
          end
          
          % Evaluate fields
          cases{1} = ProcessingParams.eval_fields(spec);
        end
      end
    end
    
    % Performs the same recursive processing as getCases, but returns
    % the number of cases rather than the array of cases.  Used for
    % finding the amount of space needed before allocating space.
    function n_cases = countCases(spec)
      spec = ProcessingParams.read_opts(spec);
      
      if iscell(spec)
        n_cases = 0;
        for k=1:numel(spec)
          n_cases = n_cases + ProcessingParams.countCases(spec{k});
        end
      elseif ~isstruct(spec)
        error('Unexpected type for spec');
      elseif numel(spec) ~= 1
        n_cases = 0;
        for k=1:numel(spec)
          n_cases = n_cases + ProcessingParams.countCases(spec(k));
        end
      else
        n_cases = 1;
        flds = fieldnames(spec);
        for k=1:length(flds)
          fld = flds{k};
          if regexp(fld,'__$','once')
            continue;
          end
          ca = spec.(fld);
          if iscell(spec.(fld)) || (isstruct(ca) && numel(ca) ~= 1)
            n_spc = 0;
            spc = spec;
            for n=1:numel(ca)
              if iscell(ca)
                spc.(fld) = ca{n};
              else
                spc.(fld) = ca(n);
              end
              n_spc = n_spc + ProcessingParams.countCases(spc);
            end
          elseif regexp(fld,'_$','once')
            spc = rmfield(spec,fld);
            ca_flds = fieldnames(ca);
            for nf = 1:length(ca_flds);
              cfld = ca_flds{nf};
              spc.(cfld) = ca.(cfld);
            end
            n_spc = ProcessingParams.countCases(spc);
          else
            continue;
          end
          n_cases = n_cases * n_spc;
          break;
        end
      end
    end
    
    % Process a list of cases with a given function
    %   Input:
    %     spec - a specfication of the a cell array of ProcessingParams
    %            objects (cases)
    %     proc_func - a function handle or a character string. If it is
    %                 function handle, this function is called for each
    %                 case, with argument case and possibly additional
    %                 arguments. If it is a character string, than
    %                 case.(proc_func) is called.
    %     proc_args - (cell array) additional arguments to proc_func
    %     options - (optional) - a struct containing additional
    %               options. The following fields are used by this
    %               function. Other fields may be present and they will
    %               be used by init().
    %       state_info - (character string) a name of a mat file which
    %              specifies the state of a previously interrupted run.
    %              If it is present, all other arguments are
    %              overridden by the values here and processing
    %              continues from where it stopped. Fields which are
    %              present in opt_args override the saved fields.
    %       parallel - replaces the default of
    %           ProcessingParams.parallel
    %       init - a handle to a function
    %              [cases, proc_func, proc_args,opt_args] = ...
    %                  init(cases, proc_func, proc_args,opt_args)
    %              which runs before execution and may change proc_args
    %              or proc_func.  Ignored if continuing after reading
    %              state_info.
    %       finish - a handle to a function
    %                results = finish(cases, results, options)
    %                which runs after execution
    %   Output:
    %     results - A cell array containing the processing results. It
    %               of the same size as getCases(specs)
    %     cases - (optional).  The result of parsing spec.
    %
    function [results, cases] = processCases(spec, proc_func, proc_args, ...
        options)
      simulation_start=tic;
      start_date = datestr(now,'yyyymmdd_HHMM');
      
      if nargin <4
        options = struct();
      end
      
      % See if continuing a previous simulation
      if isfield(options, 'state_info') && exist(options.state_info,'file')
        state_info = load(options.state_info);
        continuing = true;
        
        proc_func = state_info.proc_func;
        proc_args = state_info.proc_args;
        cases = state_info.cases;
        results = state_info.results;
        
        flds = fields(options);
        for k=1:length(flds)
          fld = flds{k};
          state_info.options.(fld) = options.(fld);
        end
        options = state_info.options;
      else
        continuing = false;
      end
      
      % Set parallel processing
      if isfield(opt_args, 'parallel')
        n_par = ceil(matlabpool('size') * opt_args.parallel);
      else
        n_par = ceil(matlabpool('size') * ProcessingParams.parallel);
      end
      
      % Initialization, if this is not a continuation case
      if ~continuing
        cases = ProcessingParams.getCases(spec);
        cases = ProcessingParams.setCaseNo(cases);
        results = cell(size(cases));
        if isfield(options, 'init') && ~isempty(options.init)
          [cases, proc_func, proc_args, options] = ...
            options.init(cases, proc_func, proc_args, options);
        end
      end
      
      % Processing
      if n_par % Parallel processing
        for bgn = 1:n_par:numel(cases)
          parfor k=bgn:bgn+n_par-1
            if ~isempty(results{k})
              continue;
            end
            if size(proc_args,1) == 1
              args = proc_args;
            else
              args = proc_args(k,:);
            end
            if ischar(proc_func)
              results{k} = cases{k}.(proc_func)(args{:});
            else
              results{k} = proc_func(cases{k}, args{:});
            end
          end
          save_state();
        end
      else % No parallel processing
        for k=1:numel(cases);
          if ~isempty(results{k})
            continue;
          end
          if size(proc_args,1) == 1
            args = proc_args;
          else
            args = proc_args(k,:);
          end
          if ischar(proc_func)
            results{k} = cases{k}.(proc_func)(args{:});
          else
            results{k} = proc_func(cases{k}, args{:});
          end
        end
        save_state();
      end
      
      if isfield(options, 'finish')
        results = options.finish(cases, results, options);
      end
      
      fprintf('\n   done. Start: %s. End: %s. %d sec.\n', start_date,...
        datestr(now,'yyyymmdd_HHMM'), toc(simulation_start));
      
      
      function save_state()
        if ~isfield(options, 'state_info')
          return
        end
        
        state_info = struct(...
          'proc_func',proc_func,...
          'proc_args',proc_args,...
          'options',options,...
          'cases',cases,...
          'options',options);
        
        save(options.state_info, '-struct', 'state_info', '-mat');
      end
    end
    
    % read_opts returns the input argument, opts, unchanged unless opts
    % is character string. In that case opts is parsed as JSON string
    % and the result is returned as a struct.  However, if opts is a
    % character string beginning with '<', itis interpreted as a path
    % to a text file containing a JSON string.  The string is read and
    % parsed as a above.
    function opts = read_opts(opts)
      if isempty(opts)
        return
      end
      if ischar(opts)
        if strcmp(opts(1),'<')
          jstr = fileread(opts(2:end));
        else
          jstr = opts;
        end
        opts = parse_json(jstr);
      end
    end
    function opts = eval_fields(opts)
      flds = fields(opts);
      for n=1:numel(opts)
        for k = 1:length(flds)
          fld = flds{k};
          val = opts(n).(fld);
          if ischar(val) && strcmp(val(1),'&')
            opts(n).(fld) = eval(val(2:end));
          elseif isstruct(val)
            % Recursive processing into sub structs
            opts(n).(fld) = ProcessingParams.eval_fields(val);
          end
        end
      end
    end
    
    % parse_opts performs the same function as read_opts, and then
    % evaluate the fields using eval_fields().
    function opts = parse_opts(opts)
      opts = ProcessingParams.read_opts(opts);
      if isstruct(opts)
        opts = ProcessingParams.eval_fields(opts);
      end
    end
  end
  
  
  methods (Abstract)
    %         str = describeParams(obj, prefix);
  end
end

