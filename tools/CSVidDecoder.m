classdef CSVidDecoder < handle
  %   CSVidDecoder Perform decoding operations
  %   Detailed explanation goes here
  
  properties (Constant)
    % Control using parallel processing on blocks. If 0 no parallel
    % processing is done.  Otherwise the maximal number of
    % blocks done in parallel is the workers pool size times this value.
    parallel_blocks = 1;
    %         parallel_blocks = 0;
  end
  
  properties
    n_parblk; % Actual number of parallel blocks
    n_parfrm; % Actual number of parallel frames
    info = struct();
    fdef;     % FilesDef object
    dec_out;  % Decoded output file
    
    % Analysis output
    ref_mark_blks = [];
    dec_mark_blks = [];
    dec_slct_mark_blks = [];
    dec_anls = [];
    dec_sav = [];
    
    total_bytes = 0;
    total_dur = 0;  % Total decoding duration (not including comparison)
    cmpr_dur = 0;   % Time spend on comparisons.
    
    % Processing options
    proc_opts = [];
    
    % Last recovered video block and motion info
    blk_motion = [];
    
    % DC value (from measurements);
    dc_val;
    
    % A CS_AnlsParams object defining decoding options.
    anls_opts;
    
    % A CS_DecParams object defining decoding options.
    solver_opts;
    
    cs_blk_data_list = {};
    cs_blk_data_cnt = 0;
    
    smbl_prfx='';
    prfx='';
    
    % Range of blocks to be processed
    blk_range = [1 1 1; inf inf inf];
    
    % A VidCompare object to reference frames and computs SNR
    vid_cmpr = [];
    
    % Number of frames to wait in writeReadyFrames
    write_frms_wait = 0;
    
    output_blks = [];
    
    dec_pre_diff_blks=[];
    pre_diff_cmpr=[];
    ref_pre_diff_frms={[],[],[]};
    ref_pre_diff_fid = -1;  % File descriptor for writing ref_pre_diff
    err_pre_diff_fid = -1;  % File descriptor for writing err_pre_diff
    
    ref_rgn_data = [];
    ref_rgn_src = [];
    
    rgn_code_len = 0;
    rgn_data = [];
    rgn_data_cnt = 0; % number of encoder regions read and pending processing
    rgn_blk_cnt = 0;  % number of blocks read and pending processing
    max_rgn_blk_cnt = 0;
    
    frms_start_time;
    frms_bytes=0;
    frms_done=0;
    frms_report=true;
    
    % Number of frames to wait in writeReadyFrames or in whole frame
    % processing
    frms_wait_cnt = 0;
    
    read_start_time;
    
    sns_mtrx_mgr;
    
    % Information needed for whole frames decoding. We assume that all
    % regions cover the same amount of frames
    % The fields in the struct are:
    %   tblk_map - a 3 D array representing block status. If the block has
    %       been read and is available for processing it is the index of
    %       the block's data in obj.rgn_data (note that more than one block
    %       can share the same entry in obj.rgn_data).  Otherwise the value
    %       is 0.
    %   max_tblks - temporal number of blocks in tblk_map
    %   tblk_ofst - temporal ofst to index of the first block wf.tblk_map.
    %   tblk_max_cnt - force processing beginning when the number of t_blks
    %     excceeds this number
    wf = struct();
    
    % A cell array of information about regions waiting for decoding
    wf_tblk_indx=zeros(0,2);    % min and max temporal block indices in region
    wf_n_tblks=[];       % no. of temporal blocks in a region.
    wf_n_rgns=[];         % no of regions found so far in the frame
    
    % A collection of sparsifiers and matrices that have been computed in
    % previous blocks. This is done in order not t repeat the computation
    % in each block. This is a struct array with fields:
    %   pos - the position of the region, as an array of entries of
    %         (FST,MID,LST). This is the key for matching new regions.
    %   sprsr - sparsifier object
    %   sns_xpnd - the expander of the sensing matrix
    sparser_list = struct('pos',cell(27,1),'sprsr',cell(27,1), ...
      'sns_xpnd',cell(27,1));
    n_sparser_list = 0;  % no. of sparsers in sparser list
    
  end
  
  properties(Access = private)
    % Cleanup items
    mex_context_clnp = [];
  end
  
  methods
    function obj = CSVidDecoder(files_def, anls_opts, slv_opts, proc_opts)
      % Constructor
      % Input:
      %   files_def - A struct defining various files.
      %   anls_opts - If missing or empty no measurements analysis is done.
      %             Otherwise it speciifies options for measurements
      %             analysis. can be an object of type CS_AnlsParams or
      %             something which can be used as an argument to construct
      %             such an object:
      %              A struct in which each field specify the property to
      %                 be changed.  An empty may be
      %                 used if no change to the defaults is necessary.
      %              A JSON string specifying such a struct.  If a field
      %                 value is a string beginning with an ampersand (&),
      %                 the field minus the ampersand prefix is evaluated
      %                 before assignment.
      %              A string containing an '<' followed by a file name. A
      %                JSON string is read from the file and converted to a
      %                struct as above.
      %   slv _opts - an object of type CS_DecParams or something which can
      %                be used as an argument to construct such an object:
      %              A struct in which each field specify the property to be
      %                 changed.  An empty struct or empty array may be used
      %                 if no change to the defaults is necessary.
      %              A JSON string specifying such a struct.  If a field
      %                 value is a string beginning with an ampersand (&),
      %                 the field minus the ampersand prefix is evaluated
      %                 before assignment.
      %              A string containing an '<' followed by a file name. A
      %                JSON string is read from the file and converted to a
      %                struct as above.
      %     proc_opts - (optional) a struct with whose fields which impact
      %               the processing mode. Possible fields values and 
      %               defaults are:
      %       prefix - (']') prefix to prepend to messagees
      %       par_blks - (CSVidDecoder.parallel_blocks) If non zero blocks will
      %             be parallel processed in groups, where each group
      %             contains par_blks whole frames.
      %       output   - If present overrides the decoded video output specified
      %              in files_def
      %       report_frm - (true) Report when frms are completed
      %       report_blk - (true) report completion of each block
      %       frms_wait_cnt - (0) in processing whole frames, maximum number
      %           of frames to wait if blks are received out of order.
      %       use_gpu - (false) If true use GPU if present.
      %       use_single - If true, uss single precision (default = false)
      
      obj.mex_context_clnp = initMexContext();
      
      if nargin < 4
        obj.setProcOpts();
      else
        obj.setProcOpts(proc_opts);
      end
      
      pool = gcp('nocreate');
      if isempty(pool)
        obj.n_parblk = 0;
        obj.n_parfrm = 0;
      else
        obj.n_parblk = obj.proc_opts.par_blks * pool.NumWorkers;
        obj.n_parfrm = pool.NumWorkers;
      end
      obj.frms_report = obj.proc_opts.report_frm;
      obj.fdef = files_def;
      
      if nargin < 2
        obj.anls_opts = CS_AnlsParams();
      elseif ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
        obj.anls_opts = CS_AnlsParams(anls_opts);
      else
        obj.anls_opts = anls_opts;
      end
      
      if nargin < 3
        obj.solver_opts = CS_DecParams();
      elseif ~isempty(slv_opts) && ~isa(slv_opts,'CS_DecParams')
        obj.solver_opts = CS_DecParams(slv_opts);
      else
        obj.solver_opts = slv_opts;
      end
      
      % Setting analysis output
      if isfield(obj.proc_opts, 'output')
        obj.dec_out = obj.proc_opts.output;
      elseif isfield(obj.fdef, 'output')
        obj.dec_out = obj.fdef.output;
      else
        obj.dec_out = [];
      end
      
      obj.frms_start_time = tic;
    end
    
    function delete(obj)
      if obj.ref_pre_diff_fid ~= -1
        fclose(obj.ref_pre_diff_fid);
      end
      if obj.err_pre_diff_fid ~= -1
        fclose(obj.err_pre_diff_fid);
      end
      
    end
    
    function setBlkRange(obj, rng)
      obj.blk_range = rng;
    end
    
    function setReadStartTime(obj)
      obj.read_start_time = tic;
    end
    
    % Initialize comparison of pre_diff recovered video to the original
    % pre-diff video.
    % Input:
    %   obj:  This object
    %   pxmx: Maximal pixel value;
    %   ref - Can be one of the following:
    %         * A RawVidInfo object describing a source file
    %         * A string, which is interpreted as the name of file
    %           containing JSON string with the information about the
    %           raw video file.
    %         * A cell array, containging frames for later use
    %   skip - (optional) number of frames to skip (default=0).
    %          Relevant only when ref is a string (file name)
    function initRefPreDiff(obj, pxmx, ref, skip)
      if ~any(obj.info.enc_opts.blk_pre_diff)
        return
      end
      if nargin < 5
        skip = 0;
      end
      
      if obj.info.enc_opts.sav_levels
        intrplt = pow2(obj.info.enc_opts.sav_levels);
      else
        intrplt = (obj.solver_opts.expand_level>VidBlocker.BLK_STT_RAW);
      end
      
      obj.pre_diff_cmpr = VidCompare(pxmx, ...
        obj.solver_opts.cmpr_frm_by_frm, ref, skip, intrplt);
      
      if isfield(obj.fdef, 'dec_ref_diff')
        [obj.ref_pre_diff_fid, err_msg] = ...
          fopen(obj.fdef.dec_ref_diff, 'w');
        if obj.ref_pre_diff_fid == -1
          error('%s failed opening dec_ref_diff file %s (%s)',...
            obj.prfx, obj.fdef.dec_ref_diff, err_msg);
        end
      end
      
      if isfield(obj.fdef, 'err_pre_diff')
        [obj.err_pre_diff_fid, err_msg] = ...
          fopen(obj.fdef.err_pre_diff, 'w');
        if obj.err_pre_diff_fid == -1
          error('%s failed opening err_pre_diff file %s (%s)',...
            obj.fdef.err_pre_diff, err_msg);
        end
      end
    end
    
    function addRefPreDiff(obj, ref)
      obj.pre_diff_cmpr.addRef(ref);
      if obj.ref_pre_diff_fid ~= -1 || obj.err_pre_diff_fid ~= -1
        for k=1:length(ref)
          if isempty(obj.ref_pre_diff_frms{k})
            obj.ref_pre_diff_frms{k} = ref{k};
          else
            obj.ref_pre_diff_frms{k} = ...
              cat(3, obj.ref_pre_diff_frms{k}, ref{k});
          end
        end
      end
    end
    
    function setPrefix(obj,prefix)
      if nargin < 2
        prefix = obj.smbl_prfx;
      else
        obj.smbl_prfx = prefix;
      end
      
      if isfield(obj.info, 'enc_opts')
        obj.prfx = obj.info.enc_opts.idStr(obj.smbl_prfx);
      else
        obj.prfx = prefix;
      end
    end
    
    % Decode a file
    %   Input:
    %     obj - this object
    %     inp - CodeSource object to read from. If this is a string it is
    %           assumed to be an input file name
    %     ref_video - (optional) reference video source to compare with
    %                 the decoded video and compute PSNR. ref_video can
    %                 be one of:
    %                 * A  RawVidInfo object describing a source file
    %                 * A file name (JSON description of the file)
    %                 * A struct with fields 'fname' for the file name
    %                   and 'skip' for the number of frames to skip.
    %     proc_opts - (optional) a struct with whose fields override the
    %       values of obj.proc_opts.
    %   Output:
    %     cs_vid_io - a CSVideoCodecInputOutputData object which defines
    %             the input parameters and returns the output of the simulation.
    
    function cs_vid_io = run(obj, inp, proc_opts)
      if nargin >=  3
        obj.setProcOpts(proc_opts);
      else
        obj.setProcOpts();
      end
      
      obj.setReadStartTime();
      
      if ischar(inp)
        input = CodeSourceFile(inp);
      else
        input = inp;
      end
      
      n_rgn_processed = 0;
      
      all_done = false;
      while ~all_done
        [code_elmnt, len, obj.info] = ...
          CodeElement.readElement(obj.info, input);
        if ischar(code_elmnt)
          exc = MException('CSVidDecoder:run',...
            '%s Error in CodeElement:readElement(): %s',...
            obj.proc_opts.prefix, code_elmnt);
          throw(exc);
        elseif (isa(code_elmnt, 'CSVidFile') && isempty(code_elmnt.name)) ||...
            (isscalar(code_elmnt) && code_elmnt == -1)
          rdata = obj.computeRdata(true); % Compute anything left over
          all_done = true;
        else
          obj.total_bytes = obj.total_bytes + len;
          rdata = obj.decodeItem(code_elmnt, len);
        end
        
        if isnumeric(rdata) && isscalar(rdata) && rdata == -1
          break;
        end
        
        if ischar(rdata)
          error('%s %s', obj.proc_opts.prefix, rdata);
        end
        
        for k=1:length(rdata)
          dec_data = rdata{k};
          
          n_rgn_processed = n_rgn_processed + 1;
          dec_data.info.do_encode = true;
          bdata = CSVideoBlockProcessingData(dec_data, dec_data.dc_val);
          obj.cs_blk_data_cnt = obj.cs_blk_data_cnt + 1;
          obj.cs_blk_data_list{obj.cs_blk_data_cnt} = bdata;
          
          ttl_dur = dec_data.read_dur + dec_data.proc_dur +...
            dec_data.write_dur;
          
          if dec_data.proc_opts.report_blk
            if isfield(dec_data, 'blk_psnr')
              psnr_str = sprintf(' PSNR=%4.1f', dec_data.blk_psnr);
            else
              psnr_str = '';
            end
            fprintf('%s blk %s. Dur: %6.2f(%6.2f) sec.%s\n', ...
              dec_data.prefix,...
              show_str(dec_data.info.vid_region.blk_indx(1,:)),...
              ttl_dur, dec_data.proc_dur, psnr_str);
          end
          obj.total_dur = obj.total_dur + ttl_dur;
        end
      end
      
      if ~isempty(obj.output_blks)
        [nfrm, vid] = obj.output_blks.writeReadyFrames(0);
        if ischar(nfrm)
          error('Failed writing decoded frames (%s)', nfrm);
        end
        if nfrm > 0 && ~isempty(obj.vid_cmpr)
          obj.vid_cmpr.update(vid, obj.prfx);
        end
      end
      
      if ~isempty(obj.dec_slct_mark_blks)
        err_msg = ...
          obj.dec_slct_mark_blks.writeReadyFrames(0);
        if ischar(err_msg);
          error('failed writing dec_slct_mark frames: %s', err_msg);
        end
      end
      
      if ~isempty(obj.dec_mark_blks)
        err_msg = ...
          obj.dec_mark_blks.writeReadyFrames(0);
        if ischar(err_msg);
          error('failed writing dec_mark frames: %s', err_msg);
        end
      end
      
      if ~isempty(obj.ref_mark_blks)
        err_msg = ...
          obj.ref_mark_blks.writeReadyFrames(0);
        if ischar(err_msg);
          error('failed writing ref_mark frames: %s', err_msg);
        end
      end
      
      if proc_opts.no_stats
        cs_vid_io = 'no_stats';
      elseif nargout > 0 
        if isfield(obj.info,'enc_opts') && isfield(obj.info,'raw_vid')
          cs_vid_io = CSVideoCodecInputOutputData(obj.info.enc_opts,...
            obj.info.raw_vid);
          cs_vid_io.calculate(obj.cs_blk_data_list(1:obj.cs_blk_data_cnt));
          [cs_vid_io.msqr_err, cs_vid_io.psnr] = obj.getFinalPSNR();
        else
          cs_vid_io = [];
        end
      end
    end
    
    % Read an encoded file while convert the format of the lossless coding
    % and write it out.
    %   Input arguments:
    %     obj - this decoder
    %     params - a struct that may contain the following fields:
    %       lossless_coder
    %       lossless_coder_AC_gaus_thrsh
    %     csv_io_in - a CSVideoCodecInputOutputData object, or the name of
    %              a file from which it can be read.\
    %     csv_io_out - a name of a file to write the modified csv_io
    %                  into
    %     vid_in - file name of the input encoded file
    %     vid_out - file name of the output encoded file
    %   Output arguments:
    %     cs_vid_io - updated value of csv_io
    function cs_vid_io = convertLLC(obj, prefix, params, csv_io_in, csv_io_out, ...
        vid_in, vid_out)
      if ischar(csv_io_in)
        csv_io_ld = load(csv_io_in);
        cs_vid_io = csv_io_ld.cs_vid_io;
      else
        cs_vid_io = csv_io_in;
      end
      
      params_flds = fieldnames(params);
      if ~isempty(params_flds)
        for fldc=params_flds
          fld = fldc{1};
          cs_vid_io.(fld) = params.(fld);
        end
      end
      
      input = CodeSourceFile(vid_in);
      
      [output_dir, output_name, output_ext] = fileparts(vid_out);
      [success, msg, ~] = mkdir(output_dir);
      if ~success
        error('failed to create directory %s (%s)', output_dir, msg);
      end
      output = CodeDest.constructCodeDest(vid_out);
      obj.rgn_code_len = 0;
      n_vid_blks_processed = 0;
      out_info = struct();
      
      elmnts = cell(10,1);
      n_elmnts = 0;
      while true
        [code_elmnt, ~, obj.info] = ...
          CodeElement.readElement(obj.info, input);
        if ischar(code_elmnt)
          error('%s CSVidDecoder:run',...
            ['Error in CodeElement:readElement(): '...
            prefix, code_elmnt]);
        elseif isscalar(code_elmnt) && code_elmnt == -1
          break;
        end
        
        n_elmnts = n_elmnts+1;
        elmnts{n_elmnts} = code_elmnt;
        
        if isa(code_elmnt, 'QuantMeasurements')
          bx = obj.info.vid_region.blk_indx;
          if ~all(bx >= ones(size(bx,1),1)*obj.blk_range(1,:))
            continue;
          end
          if ~all(bx <= ones(size(bx,1),1)*obj.blk_range(2,:))
            break;
          end
          
          obj.info.q_msr = code_elmnt;
          switch(cs_vid_io.lossless_coder)
            case cs_vid_io.LLC_INT
              out_info.q_msr = QuantMeasurementsBasic(code_elmnt);
            case cs_vid_io.LLC_AC
              out_info.q_msr = QuantMeasurementsAC(code_elmnt);
          end
          elmnts{n_elmnts} = out_info.q_msr;
          
          [len_elmnt, out_info] = CodeElement.writeElements(out_info,...
            elmnts(1:n_elmnts), output);
          if ischar(len_elmnt)
            error('writing elemnts failed: %s', len_elmnt);
          end
          n_elmnts = 0;
          obj.rgn_code_len = obj.rgn_code_len + len_elmnt;
          
          n_vid_blks_processed = n_vid_blks_processed + 1;
          bdata = CSVideoBlockProcessingData(out_info);
          obj.cs_blk_data_cnt = obj.cs_blk_data_cnt + 1;
          obj.cs_blk_data_list{obj.cs_blk_data_cnt} = bdata;
          
          fprintf('%s blk %s done.\n', prefix,...
            int2str(out_info.vid_region.blk_indx(1,:)));
          
          obj.rgn_code_len = 0;
          
        elseif isa(code_elmnt, 'VidRegion')
          obj.info.vid_region = code_elmnt;
          out_info.vid_region = code_elmnt;
        elseif isa(code_elmnt, 'CS_EncVidParams')
          obj.info.enc_opts = code_elmnt;
          obj.info.Yblk_size = code_elmnt.blk_size;
          obj.initBlocker();
          out_info.enc_opts = code_elmnt;
          out_info.Yblk_size = code_elmnt.blk_size;
          params_flds = fieldnames(params);
          for i_fld = 1:length(params_flds)
            fld = params_flds{i_fld};
            obj.info.enc_opts.(fld) = params.(fld);
            out_info.enc_opts.(fld) = params.(fld);
          end
        elseif isa(code_elmnt, 'RawVidInfo')
          obj.info.raw_vid = code_elmnt;
          out_info.raw_vid = code_elmnt;
          obj.initBlocker();
        else
          error('Unexpected object of type %s', class(code_elmnt));
        end
      end
      [len_elmnt, ~] = CodeElement.writeElements(out_info,...
        elmnts(1:n_elmnts), output);
      if ischar(len_elmnt)
        error('writing elemnts failed: %s', len_elmnt);
      end
      
      cs_vid_io.calculate(obj.cs_blk_data_list(1:obj.cs_blk_data_cnt),1);
      
      save(csv_io_out, 'cs_vid_io', '-mat');
      
      fprintf('%s file %s done\n', prefix, [output_name output_ext]);
      
    end
    
    % decodeItem reads one item from input and processes it.
    % Input:
    %   obj - this object
    %   item - a CodeElmnt object - the item to decode
    %   len - length of encoded item (bytes).
    % Output:
    %   rdata - normally empty. A cell array of dec_data structs, each
    %          containing all then information about a region. returns
    %          -1 on end of data and a character string describing an
    %          error in case of errors.
    %
    function rdata = decodeItem(obj, item, len)
      rdata = [];
      
      obj.rgn_code_len = obj.rgn_code_len + len;
      
      if isa(item, 'QuantMeasurements')
        bx = obj.info.vid_region.blk_indx;
        if ~all(bx <= ones(size(bx,1),1)*obj.blk_range(2,:))
          return;
        end
        if ~all(bx >= ones(size(bx,1),1)*obj.blk_range(1,:))
          return;
        end
        
        obj.info.q_msr = item;
        
        dec_data = struct(...
          'prfx', obj.prfx,...
          'info', obj.info,...
          'proc_opts', obj.proc_opts,...
          'anls_opts', obj.anls_opts,...
          'solver_opts', obj.solver_opts,...
          'vid_region', obj.info.vid_region,...
          'blk_sens_mtrcs', [],...
          'q_msrs', item,...
          'ref_blks', [],...
          'dc_val',[],...
          'blk_motion', [],...
          'has_motion', false(size(obj.info.vid_region.blk_indx,1),1),...
          'slct_reconstruct', false(size(obj.info.vid_region.blk_indx,1),1),...
          'reconstruction_needed', [],...
          'pre_diff_blks', ~isempty(obj.dec_pre_diff_blks),...
          'dec_blks', ~isempty(obj.output_blks),...
          'dec_slct_mrk_blks', ~isempty(obj.dec_slct_mark_blks),...
          'dec_mrk_blks', ~isempty(obj.dec_mark_blks),...
          'ref_mrk_blks', ~isempty(obj.ref_mark_blks),...
          'analysis_needed', [],...
          'msrs_len', len,...
          'blk_len',obj.rgn_code_len,...
          'read_dur', 0,...
          'proc_dur', 0,...
          'write_dur', 0 ...
          );
        
        obj.rgn_code_len = 0;
        
        dec_data.read_dur = toc(obj.read_start_time);
        
        dec_data.analysis_needed = ...
          ~isempty(obj.dec_anls) || ~isempty(obj.dec_mark_blks) ||...
          ~isempty(obj.dec_slct_mark_blks) || ~isempty(obj.ref_mark_blks);
        
        dec_data.reconstruction_needed = dec_data.dec_blks || ...
          dec_data.pre_diff_blks || dec_data.dec_mrk_blks ;

        if dec_data.reconstruction_needed
          dec_data.slct_reconstruct = true;
        end
        
        dec_data.prefix = sprintf('%s %s] ', obj.prfx, ...
          show_str(dec_data.info.vid_region.blk_indx(1,:)));
        
        if dec_data.analysis_needed
          dec_data.blk_motion = cell(size(obj.info.vid_region.blk_indx,1),1);
          if obj.info.enc_opts.par_blks >0
            dec_data.blk_sens_mtrcs = arrayfun(...
              @(iblk) { obj.sns_mtrx_mgr.getBlockMtrx(...
              dec_data.vid_region.blk_indx(iblk,:), ...
              dec_data.vid_region.getSingleBlk(iblk)) }, ...
              (1:dec_data.vid_region.n_blk)');
          else
             dec_data.blk_sens_mtrcs = { obj.sns_mtrx_mgr.getBlockMtrx(...
               dec_data.vid_region.blk_indx, dec_data.vid_region) };
          end
        end
        
        if (~isempty(obj.anls_opts) && ~isempty(obj.ref_mark_blks)) ||...
            (~isempty(obj.solver_opts) && ...
            (obj.solver_opts.init==-1 ||...
            obj.solver_opts.cmpr_blk_by_blk))
          dec_data.ref_blks = obj.getRefRegion();
          obj.ref_rgn_data = [];
        end
        
        obj.rgn_blk_cnt = obj.rgn_blk_cnt + ...
          size(obj.info.vid_region.blk_indx,1);
        obj.rgn_data_cnt = obj.rgn_data_cnt + 1;
        
        if obj.solver_opts.whole_frames
          % find an empty slot in obj.rgn_data
          for k=1:length(obj.rgn_data)
            if isempty(obj.rgn_data{k})
              indx = k;
              break;
            end
          end
          obj.rgn_data{indx} = dec_data;
          obj.wf.tblk_map = obj.info.vid_region.markBlkMap(...
            obj.wf.tblk_map, indx, obj.wf.tblk_ofst);
        else
          obj.rgn_data{obj.rgn_data_cnt} = dec_data;
        end
        
        if obj.rgn_blk_cnt >= obj.max_rgn_blk_cnt
          rdata = obj.computeRdata(false);
        end
        
        obj.info = rmfield(obj.info,'vid_region');
        obj.setReadStartTime();
        
      elseif isa(item, 'VidRegion')
        obj.info.vid_region = item;
        if obj.info.enc_opts.par_blks >0
          obj.info.sens_mtrx = obj.sns_mtrx_mgr.getFrmsMtrx(...
            obj.info.vid_region.blk_indx(1,3), obj.info.vid_region.blk_indx(end,3));
        else
          obj.info.sens_mtrx = obj.sns_mtrx_mgr.getBlockMtrx(item.blk_indx(1,:), item);
        end
      elseif isa(item, 'CS_EncVidParams')
        obj.info.enc_opts = item;
        obj.setPrefix();
        obj.info.enc_opts.setParams(struct(...
          'dec_opts', obj.solver_opts));
        obj.info.Yblk_size = item.blk_size;
        obj.initBlocker();
      elseif isa(item, 'RawVidInfo')
        obj.info.raw_vid = item;
        obj.initBlocker();
      else
        rdata = ['Unknown code element ', class(item)];
      end
      
    end
    
    function rdata = computeRdata(obj, do_all)
      % Do the actual computation
      %   Input:
      %     do_all - Relevant only if whole_frames is true. In this
      %              if do_all is true, all data is processed, even if
      %              not all blocks are available (this is done at end
      %              of data).
      %   Output:
      %   rdata - A cell array of dec_data structs, each
      %          containing all then information about a region.
      
      % Select which rgn_data to use
      if obj.solver_opts.whole_frames
        if ~isfield(obj.wf, 'tblk_map')
          rdata = [];
          return
        end
        
        n_tblks = size(obj.wf.tblk_map,3); % Do as many as possible
        if ~do_all
          for j=(n_tblks + 1 - obj.proc_opts.frms_wait_cnt): n_tblks
            slc = obj.wf.tblk_map(:,:,j);
            if any(slc(:) <= 0)
              n_tblks = j-1;
              break
            end
          end
        end
        rgn_indcs = zeros(numel(obj.wf.tblk_map(:,:,n_tblks)),1);
        rgn_ends = zeros(n_tblks,1);
        rgn_cnt = 0;
        for t=1:n_tblks
          for h=1:size(obj.wf.tblk_map,2)
            for v=1:size(obj.wf.tblk_map,1)
              if ~obj.wf.tblk_map(v,h,t)
                continue;
              end
              rgn_cnt = rgn_cnt+1;
              indx = obj.wf.tblk_map(v,h,t);
              rgn_indcs(rgn_cnt) = indx;
              dec_data = obj.rgn_data{indx};
              obj.wf.tblk_map = dec_data.info.vid_region.markBlkMap(...
                obj.wf.tblk_map, 0, obj.wf.tblk_ofst);
              obj.rgn_blk_cnt = obj.rgn_blk_cnt - ...
                dec_data.info.vid_region.n_blk;
            end
          end
          rgn_ends(t) = rgn_cnt;
        end
        rgn_indcs = rgn_indcs(1:rgn_cnt);
        rdata = obj.rgn_data(rgn_indcs);
        obj.rgn_data(rgn_indcs) = cell(1,length(rgn_indcs));
        obj.rgn_data_cnt = obj.rgn_data_cnt - rgn_cnt;
        n_left = size(obj.wf.tblk_map,3) - n_tblks;
        obj.wf.tblk_map(:,:,1:n_left) = ...
          obj.wf.tblk_map(n_tblks+1:n_tblks+n_left);
        obj.wf.tblk_map(:,:,n_left+1:end) = 0;
        obj.wf.tblk_ofst = obj.wf.tblk_ofst + n_tblks;
      else
        rdata = obj.rgn_data(1:obj.rgn_data_cnt);
        obj.rgn_data = cell(size(obj.rgn_data));
        obj.rgn_data_cnt = 0;
        obj.rgn_blk_cnt = 0;
      end
      
      % Analysis first
      if obj.n_parblk
        parfor k=1:length(rdata)
          rdata{k} = CSVidDecoder.analyzeDecData(rdata{k});
        end
      else
        for k=1:length(rdata)
          rdata{k} = CSVidDecoder.analyzeDecData(rdata{k});
        end
      end
      
      % Create sparsifiers
      rdata = obj.setSparsifiers(rdata);
      
      % Perform decoding
      if obj.solver_opts.whole_frames
        rgn_bgns = [1; (rgn_ends(1:end-1)+1)];
        
        frmdata = cell(n_tblks,1);
        for t = 1:n_tblks
          frmdata{t} = rdata(rgn_indcs(rgn_bgns(t):rgn_ends(t)));
        end
        if obj.n_parfrm
          parfor t = 1:n_tblks
            frmdata{t} = CSVidDecoder.jointCalculateDecData(frmdata{t});
          end
        else
          for t = 1:n_tblks
            frmdata{t} = CSVidDecoder.jointCalculateDecData(frmdata{t});
          end
        end
        for t = 1:n_tblks
          rdata(rgn_indcs(rgn_bgns(t):rgn_ends(t))) = frmdata{t};
        end
        
      else   % not whole frames
        
        if ~isempty(rdata)

          if obj.n_parblk
            parfor k=1:length(rdata)
              rdata{k} = ...
                CSVidDecoder.calculateDecData(rdata{k});
            end
          else
            for k=1:length(rdata)
              rdata{k} = ...
                CSVidDecoder.calculateDecData(rdata{k});
            end
          end
        end
      end
      
      % Writing out
      for k=1:length(rdata)
        rdata{k} = obj.writeDecData(rdata{k});
      end
    end
    
    function [rdata, mse, psnr] = finish(obj)
      dec_finish_start = tic;
      
      rdata = obj.computeRdata(true); % Compute anything left over
      
      obj.total_dur = obj.total_dur + toc(dec_finish_start);
      
      
      [mse, psnr] = obj.getFinalPSNR();
    end
    
    function [mse, psnr] = getFinalPSNR(obj)
      if ~isempty(obj.pre_diff_cmpr)
        pre_diff_psnr = obj.pre_diff_cmpr.getPSNR();
        fprintf('%s pre diff PSNR: %f\n', obj.prfx, pre_diff_psnr);
      end
      
      % If necessary run comparisons
      if ~isempty(obj.vid_cmpr) && obj.vid_cmpr.sqr_err.n_pnt
        [psnr, mse] = obj.vid_cmpr.getPSNR();
      else
        psnr = []; mse=[];
      end
    end
  end
  
  methods (Access=private)
    
    function setProcOpts(obj, proc_opts)
      % If the processing options are not defined it sets them to the
      % default. Then, if a second argument is present (should be a struct),
      % Its fields are used to override the default arguments.
      % The options and defaults are:
      %   prefix - (']') prefix to prepend to messagees
      %   par_blks - (CSVidDecoder.parallel_blocks) If non zero blocks will
      %             be parallel processed in groups, where each group
      %             contains par_blks whole frames.
      %   output   - If present overrides the decoded video output specified
      %              in files_def
      %   report_frm - (true) Report when frms are completed 
      %   report_blk - (true) report completion of each block
      %   frms_wait_cnt - (0) in processing whole frames, maximum number
      %           of frames to wait if blks are received out of order.
      %   use_gpu - (false) If true use GPU if present.
      %   use_single - If true, uss single precision (default = false)
      
      if isempty(obj.proc_opts)
        obj.proc_opts = struct(...,
          'prefix', ']',...
          'par_blks', CSVidDecoder.parallel_blocks,...
          'report_frm', true,...
          'report_blk', true,...
          'frms_wait_cnt', 0,...
          'use_gpu', CompMode.defaultUseGpu(),...
          'use_single', false,...
          'no_stats', false);
      end
      
      if nargin > 1 && ~isempty(proc_opts)
        flds = fieldnames(proc_opts);
        for k=1:length(flds)
          fld = flds{k};
          obj.proc_opts.(fld) = proc_opts.(fld);
        end
      end
      
      if obj.proc_opts.par_blks
        t = getCurrentTask();
        if ~isempty(t)
          obj.proc_opts.par_blks = 0;  % Already in a worker
        end
      end
      
      if obj.proc_opts.use_gpu && ~isfield(obj.proc_opts, 'gpus')
        proc_opts.gpus = find_gpus();
        if isempty(proc_opts.gpus)
          obj.proc_opts.use_gpu = false;
        else
        end
      end
      if ~obj.proc_opts.use_gpu && CompMode.defaultUseGpu()
        error('Specified no_gpu while MEX SW is compiled for GPU');
      end
      
      if proc_opts.use_gpu ~= CompMode.defaultUseGpu()
        error('proc_opts.use_gpu=%d while CompMode.defaultUseGpu()=%d',...
        proc_opts.use_gpu, CompMode.defaultUseGpu())
      end
      CompMode.setDefaultUseSingle(proc_opts.use_single);
    end
    
    function rdata = setSparsifiers(obj, rdata)
      prev_n_list = obj.n_sparser_list;
      for k=1:length(rdata)
        start_time = tic;
        
        dec_data = rdata{k};
        if ~dec_data.reconstruction_needed
          continue;
        end
        
        % Set sparsifier
        indx = 0;
        sprsr_pos = dec_data.info.vid_region.indexPosition();
        for m = 1:obj.n_sparser_list
          if isequal(sprsr_pos, obj.sparser_list(m).pos)
            indx = m;
            break;
          end
        end
        if ~indx
          slv_opts = obj.solver_opts;
          vdrg = dec_data.info.vid_region;
          
          sens_expndr = vdrg.getExpandMtrx(...
            slv_opts.expand_level, VidBlocker.BLK_STT_ZEXT, 1E-12);
          
          sprsr_prms = slv_opts.sparsifier.args;
          sprsr_prms.vdrg  = vdrg;
          sprs_expndr = vdrg.getExpandMtrx(...
            slv_opts.expand_level, sprsr_prms.b_stt, 1E-12);
          sprsr_prms.expndr = SensingMatrixCascade.constructCascade(...
            {sprs_expndr.M, sens_expndr.R});
          %                     sprsr_prms.wgts = abs(sprs_expndr.M.multVec(...
          %                       sens_expndr.V.multTrnspAbs(sens_expndr.L.getDiag())));
          
          indx = obj.n_sparser_list+1;
          obj.n_sparser_list = indx;
          obj.sparser_list(indx).pos = sprsr_pos;
          % Temporarily store sprsr_prms here until the
          % sparsifier is computed.
          obj.sparser_list(indx).sprsr = sprsr_prms;
          obj.sparser_list(indx).sns_xpnd = sens_expndr;
        end
        rdata{k}.sparser_list_indx = indx;
        
        rdata{k}.proc_dur = rdata{k}.proc_dur + toc(start_time);
      end
      
      if obj.n_sparser_list > prev_n_list
        % Since computing the exact norm is computationally heavy
        % we do it once for each sparser and if possible, in
        % parallel. Note that SenseMatrix is a handle object,
        % computing getExactNorm() on sm updates
        % slist(k).sprsr.
        %
        slist = obj.sparser_list(prev_n_list+1:obj.n_sparser_list);
        type = slv_opts.sparsifier.type;
        durs_list = zeros(size(slist));
        use_gpu = obj.proc_opts.use_gpu;
        use_single = obj.proc_opts.use_single;
        if obj.n_parblk
          parfor k=1:length(slist)
            start_time = tic;
            slist(k).sprsr.use_gpu = use_gpu;
            slist(k).sprsr.use_single = use_single;
            slist(k).sprsr = BaseSparser.construct(...
              type, slist(k).sprsr);
            durs_list(k) = toc(start_time);
          end
        else
          for k=1:length(slist)
            start_time = tic;
            slist(k).sprsr.use_gpu = use_gpu;
            slist(k).sprsr.use_single = use_single;
            slist(k).sprsr = BaseSparser.construct(...
              type, slist(k).sprsr);
            durs_list(k) = toc(start_time);
          end
        end
        obj.sparser_list(prev_n_list+1:obj.n_sparser_list) = slist;
      end
      
      for k=1:length(rdata)
        start_time = tic;
        if ~isfield(rdata{k}, 'sparser_list_indx')
          continue;
        end
        indx = rdata{k}.sparser_list_indx;
        if indx > prev_n_list
          idx = indx - prev_n_list;
          rdata{k}.proc_dur = rdata{k}.proc_dur + durs_list(idx);
          durs_list(idx) = 0;
        end
        rdata{k}.sparser_info = duplicate(obj.sparser_list(indx));
        
        rdata{k}.proc_dur = rdata{k}.proc_dur + toc(start_time);
      end
    end
    
    
    function dec_data = writeDecData(obj, dec_data)
      start_time = tic;
      
      obj.dc_val = dec_data.dc_val;
      
      if ~isempty(dec_data.pre_diff_blks)
        obj.writePreDiff(dec_data.pre_diff_blks);
      end
      
      obj.frms_bytes = obj.frms_bytes + double(dec_data.blk_len);
      
      % Write analysis into CSV and SAV files
      if ~isempty(obj.dec_anls) || ~isempty(obj.dec_sav)
        for k = 1:dec_data.info.vid_region.n_blk
          b_indx = dec_data.info.vid_region.blk_indx(k,:);
          dec_data.blk_motion.setBlkInfo(...
            dec_data.info.vid_blocker.getBlkInfo(b_indx));
          if ~isempty(obj.dec_anls)
            obj.dec_anls.writeRecord(...
              dec_data.blk_motion.getCSVRecord());
          end
          if ~isempty(obj.dec_sav)
            obj.dec_sav.setBlkRecord(dec_data.blk_motion);
          end
        end
      end
      
      % Write decoded files
      if ~isempty(dec_data.dec_blks)
        obj.output_blks = dec_data.info.vid_region.putIntoBlkArray(...
          dec_data.dec_blks, obj.output_blks);
        [nfrm, vid] = obj.output_blks.writeReadyFrames(obj.write_frms_wait);
        if ischar(nfrm)
          error('Failed writing decoded frames (%s)', nfrm);
        end
        if nfrm > 0
          dec_data.frms_done = nfrm;
          if obj.frms_report
            if ~isempty(obj.vid_cmpr)
              dec_data.frms_psnr = obj.vid_cmpr.update(vid, obj.prfx);
              psnr_str = sprintf(' PSNR=%.1f', dec_data.frms_psnr);
            else
              psnr_str = '';
            end
            if obj.info.enc_opts.process_color
              frms_b = 8*obj.frms_bytes/(nfrm*obj.info.raw_vid.frame_len);
            else
              frms_b = 8*obj.frms_bytes/(nfrm*obj.info.raw_vid.Ylen);
            end
            obj.frms_bytes = 0;
            frms_dur = toc(obj.frms_start_time);
            obj.frms_start_time = tic;
            fprintf('%s frames %d:%d done in %.1f sec., %.3f b/pxl %s\n',...
              obj.prfx, obj.frms_done+1, obj.frms_done+nfrm, frms_dur, ...
              frms_b, psnr_str);
            obj.frms_done = obj.frms_done + nfrm;
          else
            if ~isempty(obj.vid_cmpr)
              dec_data.frms_psnr = obj.vid_cmpr.update(vid, obj.prfx);
            end
            obj.frms_bytes = 0;
            obj.frms_start_time = tic;
          end
        end
      end
      
      if ~isempty(dec_data.dec_slct_mrk_blks)
        obj.dec_slct_mark_blks = dec_data.info.vid_region.putIntoBlkArray(...
          dec_data.dec_slct_mrk_blks, obj.dec_slct_mark_blks);
        err_msg = ...
          obj.dec_slct_mark_blks.writeReadyFrames(obj.write_frms_wait);
        if ischar(err_msg);
          error('failed writing dec_slct_mark frames: %s', err_msg);
        end
      end
      
      if ~isempty(dec_data.dec_mrk_blks) && ~isempty(obj.dec_mark_blks)
        obj.dec_mark_blks = dec_data.info.vid_region.putIntoBlkArray(...
          dec_data.dec_mrk_blks, obj.dec_mark_blks);
        err_msg = ...
          obj.dec_mark_blks.writeReadyFrames(obj.write_frms_wait);
        if ischar(err_msg);
          error('failed writing dec_mark frames: %s', err_msg);
        end
      end
      
      if ~isempty(dec_data.ref_mrk_blks)
        obj.ref_mark_blks = dec_data.info.vid_region.putIntoBlkArray(...
          dec_data.ref_mrk_blks, obj.ref_mark_blks);
        err_msg = ...
          obj.ref_mark_blks.writeReadyFrames(obj.write_frms_wait);
        if ischar(err_msg);
          error('failed writing ref_mark frames: %s', err_msg);
        end
      end
      
      dec_data.write_dur = toc(start_time);
    end
    
    function ref_rgn = getRefRegion(obj)
      if isempty(obj.ref_rgn_data) && isfield(obj.fdef, 'input')
        if isempty(obj.ref_rgn_src)
          spec = obj.info.enc_opts;
          vid_in_params = struct();
          if ~isempty(obj.solver_opts)
            vid_in_params.w_type_d = obj.solver_opts.wnd_type;
          end
          if spec.sav_levels
            intrplt = pow2(spec.sav_levels);
          else
            intrplt = 1;
          end
          obj.ref_rgn_src = VidBlocksIn(vid_in_params, obj.fdef.input,...
            obj.solver_opts.expand_level, spec, intrplt);
          obj.ref_rgn_src.use_gpu = obj.proc_opts.use_gpu;
          obj.ref_rgn_src.use_single = obj.proc_opts.use_single;
        end
        
        obj.ref_rgn_data = cell(size(obj.info.vid_region.blk_indx,1),1)';
        for k = 1:obj.info.vid_region.n_blk
          b_indx = obj.info.vid_region.blk_indx(k,:);
          blk = obj.ref_rgn_src.getBlks(b_indx);
          if isempty(blk)
            error('%s Ref block index [%s] is empty', obj.prfx,...
              int2str(b_indx));
          end
          obj.ref_rgn_data{k} = blk;
        end
        obj.ref_rgn_src.discardFrmsBeforeBlk(...
          obj.info.vid_region.blk_indx(1,:));
      end
      ref_rgn = obj.ref_rgn_data;
    end
    
    function initBlocker(obj)
      if ~isfield(obj.info, 'raw_vid')...
          || ~isfield(obj.info, 'enc_opts') || isempty(obj.info.enc_opts)
        return
      end
      
      if obj.info.enc_opts.sav_levels
        intrplt = pow2(obj.info.enc_opts.sav_levels);
      else
        intrplt = (obj.solver_opts.expand_level > VidBlocker.BLK_STT_RAW);
      end
      
      obj.info.raw_vid.setInterpolate(intrplt);
      
      % Set vid_compare if necessary
      if isfield(obj.fdef, 'input') && ~isempty(obj.solver_opts)
        obj.vid_cmpr = VidCompare(obj.info.raw_vid.getPixelMax(),...
          0, obj.fdef.input, obj.info.raw_vid.seg_start_frame, intrplt);
      end
      
      obj.initRefPreDiff(obj.info.raw_vid.getPixelMax(), ...
        obj.info.raw_vid.createEmptyVideo(0));
      
      %calculate the dimensions of the read in video
      if obj.info.enc_opts.n_frames == -1
        obj.info.enc_opts.setParams(struct('n_frames',...
          obj.info.raw_vid.n_frames - obj.info.enc_opts.start_frame + 1));
      end
      
      vid_in_params = struct();
      if ~isempty(obj.solver_opts)
        vid_in_params.w_type_d = obj.solver_opts.wnd_type;
      end
      obj.info.vid_blocker = VidBlocker(...
        vid_in_params, obj.info.raw_vid, obj.info.enc_opts);
      obj.info.vid_blocker.use_gpu = obj.proc_opts.use_gpu;
      obj.info.vid_blocker.use_single = obj.proc_opts.use_single;
      
      % Calculate number of blocks
      blk_cnt = obj.info.vid_blocker.blk_cnt;
      
      % Initialize sns_mtrx_mgr
      obj.sns_mtrx_mgr = SensingMatrixManager(obj.info.enc_opts, obj.info.vid_blocker);
      
      obj.info.quantizer = UniformQuantizer.makeUniformQuantizer(obj.info.enc_opts, ...
        obj.info.vid_blocker, obj.sns_mtrx_mgr, obj.proc_opts.use_gpu);
        
      % create region storage area
      if obj.solver_opts.whole_frames
        obj.write_frms_wait = 1;
        obj.wf.tblk_ofst = 0;
        obj.wf.max_tblks = ...
          obj.proc_opts.frms_wait_cnt+max(1,obj.n_parfrm);
        obj.wf.tblk_map = zeros(blk_cnt(1),blk_cnt(2),obj.wf.max_tblks);
        obj.max_rgn_blk_cnt = numel(obj.wf.tblk_map);
      else
        obj.write_frms_wait = obj.proc_opts.frms_wait_cnt;
        pool = gcp('nocreate');
        if isempty(pool)
          obj.max_rgn_blk_cnt = 1;
        else
          obj.max_rgn_blk_cnt = pool.NumWorkers * obj.n_parblk;
        end
      end
      obj.rgn_data = cell(1,obj.max_rgn_blk_cnt);
      
      %initialize storage for recovered video block
      obj.cs_blk_data_list = cell(blk_cnt);
      obj.cs_blk_data_list = obj.cs_blk_data_list(:);
      
      % open output files requiring analysis
      if ~isempty(obj.anls_opts)
        if isfield(obj.fdef, 'dec_anls')
          obj.dec_anls = CSVOutFile(obj.fdef.dec_anls,...
            obj.info.vid_blocker.getVidInfoFields(),...
            'video info', 1);
          obj.dec_anls.writeRecord(obj.info.vid_blocker.getVidInfo());
          obj.dec_anls.setCSVOut(BlkMotion.csvProperties(),...
            'measurements analysis');
        end
        if isfield(obj.fdef, 'dec_sav')
          obj.dec_sav = SAVOutFile(obj.fdef.dec_sav,...
            obj.info.vid_blocker);
        end
        if isfield(obj.fdef, 'input') && ...
            isfield(obj.fdef,'ref_mark') &&...
            isempty(obj.ref_mark_blks)
          obj.ref_mark_blks = VidBlocksOut(obj.info.vid_blocker, ...
            obj.fdef.ref_mark, false, obj.solver_opts.expand_level);
          obj.ref_mark_blks.use_gpu = obj.proc_opts.use_gpu;
          obj.ref_mark_blks.use_single = obj.proc_opts.use_single;
        end
        
        % open output files requiring both analysis and
        % reconstruction
        if ~ isempty(obj.solver_opts)
          if isfield(obj.fdef, 'dec_mark')
            obj.dec_mark_blks = VidBlocksOut(obj.info.vid_blocker, ...
              obj.fdef.dec_mark, false, obj.solver_opts.expand_level);
            obj.dec_mark_blks.use_gpu = obj.proc_opts.use_gpu;
            obj.dec_mark_blks.use_single = obj.proc_opts.use_single;
          end
          if isfield(obj.fdef, 'dec_slct_mark')
            obj.dec_slct_mark_blks = VidBlocksOut(obj.info.vid_blocker, ...
              obj.fdef.dec_slct_mark, false,obj.solver_opts.expand_level);
            obj.dec_slct_mark_blks.use_gpu = obj.proc_opts.use_gpu;
            obj.dec_slct_mark_blks.use_single = obj.proc_opts.use_single;
          end
        end
      end
      
      % Open output files requiring reconstruction
      if ~isempty(obj.solver_opts)
        if ~isempty(obj.dec_out)
          obj.output_blks = VidBlocksOut(obj.info.vid_blocker, ...
            obj.dec_out, false, obj.solver_opts.expand_level);
          obj.output_blks.use_gpu = obj.proc_opts.use_gpu;
          obj.output_blks.use_single = obj.proc_opts.use_single;
        end
        
        if any(obj.info.enc_opts.blk_pre_diff) &&...
            (isfield(obj.fdef, 'dec_pre_diff') ||...
            isfield(obj.fdef, 'err_pre_diff') ||...
            isfield(obj.fdef, 'dec_ref_diff'))
          
          if isfield(obj.fdef, 'dec_pre_diff')
            obj.dec_pre_diff_blks = VidBlocksOut(obj.info.vid_blocker, ...
              obj.fdef.dec_pre_diff, true, obj.solver_opts.expand_level);
          else
            obj.dec_pre_diff_blks = VidBlocksOut(obj.info.vid_blocker, ...
              [], true, obj.solver_opts.expand_level);
          end
          obj.dec_pre_diff_blks.use_gpu = obj.proc_opts.use_gpu;
          obj.dec_pre_diff_blks.use_single = obj.proc_opts.use_single;
        end
      end
    end
    
    function writePreDiff(obj,pre_diff_blks)
      obj.dec_pre_diff_blks = ...
        obj.info.vid_region.putIntoBlkArray(...
        pre_diff_blks, obj.dec_pre_diff_blks);
      [nfrm, dec_pre_diff] = ...
        obj.dec_pre_diff_blks.writeReadyFrames(obj.write_frms_wait);
      if ischar(nfrm)
        error('Failed writing dec_pre_diff frames (%s)', nfrm);
      end
      
      if ~isempty(obj.pre_diff_cmpr)  && nfrm
        obj.pre_diff_cmpr.update(dec_pre_diff, [obj.prfx ' (diff)']);
        
        if obj.ref_pre_diff_fid ~= -1 || obj.err_pre_diff_fid ~= -1
          
          if obj.ref_pre_diff_fid ~= -1
            write_pre_diff_vid(obj.ref_pre_diff_fid,...
              nfrm, obj.ref_pre_diff_frms, 'ref_pre_diff');
          end
          
          if obj.err_pre_diff_fid ~= -1
            % generate pre_diff error
            nclr = min(length(obj.ref_pre_diff_frms),...
              length(dec_pre_diff));
            err_pre_diff = cell(1,nclr);
            for iclr=1:nclr
              err_pre_diff{iclr} = dec_pre_diff{iclr} - ...
                obj.ref_pre_diff_frms{iclr}(:,:,1:nfrm);
            end
            
            % write out pre_diff error
            write_pre_diff_vid(obj.err_pre_diff_fid,...
              nfrm, err_pre_diff, 'err_pre_diff');
          end
          
          % Remove written frames
          for iclr=1:length(obj.ref_pre_diff_frms)
            obj.ref_pre_diff_frms{iclr} = ...
              obj.ref_pre_diff_frms{iclr}(:,:,nfrm+1:end);
          end
          
        end
      end
      
      function write_pre_diff_vid(fid, nfrm, vid, name)
        out_vid = obj.info.raw_vid.createEmptyVideo(nfrm);
        for kk=1:length(vid)
          out_vid{kk} = vid{kk}(:,:,nfrm);
        end
        
        err_msg = write_raw_video(fid, out_vid, obj.info.raw_vid);
        if ~isempty(err_msg)
          error('%s failed writing %s(%s)',...
            obj.prfx, name, err_msg);
        end
      end
    end
    
    
  end
  
  methods (Static)
    function [recovered_input]=recoverInputFromCSMeasurements( ...
        sens_mtrx, ...
        sparser, compressed_sensed_measurements, use_old)
      % (Decoder) runs a l1 minimization algorithm to recover an approximation
      % of the input from compressed sensed measurements
      %
      % [Input]
      %
      % indices_of_WH_cofficients_to_save - which indices of the
      %  Walsh-Hadamard transform coefficent were saved (specifies the row
      %  shuffling + selection of only top K (number of compressed sensed
      %  measurements) rows of the square
      %  Walsh-Hadamard matrix)
      %
      % permutation_of_input -  a list of numbers that represents a permuation of
      % the indices of the input. This list represents the shuffling of the
      % columns of the square Walsh-Hadamard matrix
      %
      % compressed_sensed_measurements - the compressed sensed measurements
      % generated from a fat, row and column shuffled Walsh-Hadamard
      % transform matrix
      %
      % height_of_original_input - height of the original input (before
      % vectorization)
      %
      % width_of_original_input - width of the original input (before
      % vectorization)
      %
      % frames_of_original_input - franes(temporal) of the original input (before
      % vectorization)
      %
      % [Output]
      %
      % recovered_input - the input recovered by the l1 minimization
      % algorithm
      %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      % set options for TVAL3_CVS
      clear opts
      opts.mu = 2^6;               % correspond to ||Au-b||^2
      opts.beta = 2^2;             % correspond to ||Du-w||^2
      opts.mu0 = 4;
      opts.beta0 = 1/4;
      opts.beta_A = opts.mu;
      opts.beta_D = opts.beta;
      opts.beta_A0 = opts.mu0;
      opts.beta_D0 = opts.beta0;
      opts.beta_rate_ctn = 2;
      opts.beta_rate_thresh = 0.2;
      opts.tol = 1e-4; %1e-4
      opts.tol_inn = 1e-3; %1e-3
      opts.maxin = 12;
      opts.maxcnt = 12;
      opts.StpCr = 0;              % 0 (optimality gap) or 1 (relative change)
      opts.disp = 150;
      opts.aqst = true;
      opts.isreal = true;
      opts.init = 0;
      
      %run recovery algorithm
      
      switch use_old
        case 1
          [recovered_input, ~] = TVAL3_CVS_D2_a(sens_mtrx, sparser,...
            double(compressed_sensed_measurements), opts);
        case 2
          bi = sparser.vid_region.blk_size(1,:);
          [recovered_input, ~] = TVAL3_CVS_D2(...
            sens_mtrx.getHandle_multLR(), ...
            double(compressed_sensed_measurements), bi(1),bi(2),bi(3),opts);
      end
      
      recovered_input = sparser.vid_region.pixelize(recovered_input, ...
        sparser.blk_stt);
      
    end
    
    function cs_vid_io = doSimulationCase(anls_opts, dec_opts, files_def,...
        proc_opts)
      % Run decoding simulation on a multiple files in one condition
      %   Input
      %     anls_opts - If missing or empty no measurements analysis is done.
      %             Otherwise it speciifies options for measurements
      %             analysis. can be an object of type CS_AnlsParams or
      %             something which can be used as an argument to construct
      %             such an object:
      %              A struct in which each field specify the property to
      %                 be changed.  An empty may be
      %                 used if no change to the defaults is necessary.
      %              A JSON string specifying such a struct.  If a field
      %                 value is a string beginning with an ampersand (&),
      %                 the field minus the ampersand prefix is evaluated
      %                 before assignment.
      %              A string containing an '<' followed by a file name. A
      %                JSON string is read from the file and converted to a
      %                struct as above.
      %     dec_opts - an object of type CS_DecParams or something which
      %         can be used as an argument to construct such an object:
      %           A struct in which each field specify the property to be
      %             changed.  An empty struct or empty array may be used if
      %             no change to the defaults is necessary.
      %           A JSON string specifying such a struct.  If a field value is
      %             a string beginning with an ampersand (&), the field minus
      %             the ampersand prefix is evaluated before assignment.
      %           A string containing an '<' followed by a file name. A JSON
      %             string is read from the file and converted to a struct as
      %             above.
      %     files_def - A FilesDef object, or a JSON string defining such
      %           an object, or a string bgining with '<' which defines a
      %           file from which the JSON string can be read.
      %     proc_opts (Optional) - a struct with any of the following
      %         fields which impact the processing mode (other fields are
      %         ignored):
      %         io_types - (optional) cell array of types to be processed.
      %         output_id - (optional) If present, the output directory is
      %                     set to output_id.  If output_id contains an
      %                     sterisk, the sterisk is replaced by a
      %                     date.
      %         dec_id - Decoded output directory name. Inserted into dec_dir
      %                 in io_def. If it contains an asterisk the asterisk is
      %                 replaced by a time-stamp.
      %         io_types - an optional cell array of the types to be
      %             processed (used as the optional argument fields in
      %             FilesDef constructor). This argument is ignored if
      %             io_def is a FilesDef object.
      %         case_dir - the case directory name (forces this name)
      %         prefix - (0ptional) and identifying prefix to add
      %                  before all printing. Default: '] '
      %         par_files - If true process files in parallel.
      %                     Default: false.
      %         blk_rng - If present specifies blocks range for
      %                   processing. A 2x3 array, upper row is minimum,
      %                   lower row is maximum
      %         blk_report - If true, report completion of each block
      %         use_gpu - If true, use GPU (default = false).
      %         use_single - If true, uss single precision (default = false)
      %         no_stats - If present and true, statistics are not computed,
      %         prof_spec - profiling specification. allowed values are:
      %                       0 - no profiling
      %                       1 - profile only run() function.
      %                       2 - profile everything.
      %                     default: 0
      %    Output
      %       cs_vid_io - a CSVideoCodecInputOutputData object which defines
      %             the input parameters and returns the output of the
      %             simulation. If proc_opts.no_stat is true, the string
      %             "no_stats" is returned instead
      
      case_start = tic;
      
      if nargin >= 4
        prof_ctrl(1, proc_opts);
      else
        prof_ctrl(1);
      end
    
      def_proc_opts = struct(...
        'dec_id', '*',...
        'prefix', '] ',...
        'par_files', false,...
        'use_gpu', CompMode.defaultUseGpu(),...
        'use_single', false,...
        'no_stats', false,...
        'prof_spec', 0);
      
      if nargin >= 4
        flds = fieldnames(proc_opts);
        for k=1:length(flds)
          fld = flds{k};
          def_proc_opts.(fld) = proc_opts.(fld);
        end
      end
      proc_opts = def_proc_opts;
      
      if proc_opts.use_gpu && ~isfield(proc_opts, 'gpus')
        proc_opts.gpus = find_gpus();
        if isempty(proc_opts.gpus)
          proc_opts.use_gpu = false;
        end
      end
      
      io_params = struct('identifier', struct());
      if isfield(proc_opts, 'output_id')
        io_params.identifier.output = proc_opts.output_id;
      end
      if isfield(proc_opts, 'dec_id')
        io_params.identifier.dec = regexprep(proc_opts.dec_id, '*',...
          datestr(now,'yyyymmdd_HHMM'),'once');
      end
      if isfield(proc_opts, 'io_types')
        io_params.fields = proc_opts.io_types;
      end
      prefix = proc_opts.prefix;
      
      if isa(files_def, 'FilesDef')
        files_def.specifyDirs(io_params.identifier);
      else
        files_def = FilesDef(files_def, io_params);
      end
      if isfield(proc_opts, 'case_dir')
        files_def.setDirs(struct('case', proc_opts.case_dir));
      end
      
      svn_ver = getSvnVersion();
      mex_clnp = initMexContext();  %#ok
      
      fprintf('%sStarting simulation case. SW version: %s\n%s output directory=%s \n', ...
        prefix, svn_ver, prefix, files_def.getDecoderDir());
      
      files_def.makeDirs();
      dec_dir = files_def.getDecoderDir();
      
      % Print file specification info
      mat2json(struct('files_specs',files_def.getAllFiles()),...
        fullfile(dec_dir,'files_specs.json'));
      
      write_str_to_file(dec_dir, 'version.txt', svn_ver);
      if ~isempty(anls_opts)
        if ~isa(anls_opts,'CS_AnlsParams')
          anls_opts = CS_AnlsParams(anls_opts);
        end
        fprintf('%s Analysis options\n', prefix);
        fprintf('%s\n',anls_opts.describeParams(prefix));
        
        write_str_to_file(dec_dir, 'anls_opts.txt', anls_opts.describeParams());
        anls_opts.getJSON(fullfile(dec_dir, 'anls_opts.json'));
      else
        fprintf('%s No analysis\n', prefix);
      end
      
      if ~isempty(dec_opts)
        if ~isa(dec_opts, 'CS_DecParams')
          dec_opts = CS_DecParams(dec_opts);
        end
        fprintf('%s Decoding options\n', prefix);
        fprintf('%s\n',dec_opts.describeParams(prefix));
        
        write_str_to_file(dec_dir, 'dec_opts.txt', dec_opts.describeParams());
        dec_opts.getJSON(fullfile(dec_dir, 'dec_opts.json'));
      else
        fprintf('%s No decoding', prefix);
      end
      
      proc_opts_str = show_str(proc_opts, struct(),...
        struct('prefix',prefix, 'struct_marked', true));
      fprintf('%s proc_opts:\n%s\n', prefix, proc_opts_str);
      write_str_to_file(dec_dir, 'proc_opts.txt', show_str(proc_opts, struct(),...
        struct('prefix','', 'struct_marked', true)));
      mat2json(proc_opts, fullfile(dec_dir, 'proc_opts.json'));
      
      if ~proc_opts.no_stats
        cs_vid_io = CSVideoCodecInputOutputData();
        cs_vid_io.clearResults();
      else
        cs_vid_io = 'no_stats';
      end
      
      if isempty(files_def.sock)
        par_files = proc_opts.par_files;
      else
        par_files = 0;
      end
      
      if ~isempty(proc_opts.gpus)
        init_gpu(proc_opts.gpus(1));
      end
      
      if par_files
        fdef_list = files_def.getAllFiles();
        results = cell(size(fdef_list));
        dec_opts_fdef = cell(size(fdef_list));
        anls_opts_fdef = cell(size(fdef_list));
        proc_opts_fdef = cell(size(fdef_list));
        for k=1:length(fdef_list)
          dec_opts_fdef{k} = dec_opts;
          anls_opts_fdef{k} = anls_opts;
          proc_opts_fdef{k} = proc_opts;
          proc_opts_fdef{k}.prefix = sprintf('%s%d]', prefix, k);
        end
        parfor k=1:length(fdef_list)
          fldef = fdef_list(k);
          anls_def = anls_opts_fdef{k};
          dec_def = dec_opts_fdef{k};
          proc_def = proc_opts_fdef{k};
          if isequal(dec_def.init, -1)
            dec_def.ref = fldef.input;
          end
          dec = CSVidDecoder(fldef, anls_def, dec_def, proc_def);
          if isfield(proc_opts, 'blk_rng')
            dec.setBlkRange(proc_opts.blk_rng);
          end
          
          prof_ctrl(2, proc_def.prof_spec);
        
          result= dec.run(fldef.enc_vid, proc_def);

          prof_ctrl(1, proc_def.prof_spec);
        
          if ~isempty(result)
            if ~isempty(result.psnr)
              psnr_str = sprintf('PSNR=%f4.1dB ', result.psnr);
            else
              psnr_str = '';
            end
            fprintf('%s file(%d) %s ==>\n  %s\n     %sDur. %6.2f sec.\n',...
              proc_def.prefix, k, fldef.enc_vid, fldef.output, ...
              psnr_str, dec.total_dur);
          end
          results{k} = result;
        end
        
        if ~proc_opts.no_stats
          first_result = 0;
          for k=1:length(fdef_list)
            if ~isempty(results{k})
              if ~first_result
                first_result = k;
              end
              cs_vid_io.add(results{k});
            end
          end
          if first_result
            cs_vid_io.setParams(struct(...
              'msrmnt_input_ratio', results{first_result}.msrmnt_input_ratio,...
              'qntzr_ampl_stddev', results{first_result}.qntzr_ampl_stddev,...
              'qntzr_wdth_mltplr', results{first_result}.qntzr_wdth_mltplr));
          end
        end
      else
        k=1;
        if isempty(files_def.sock)
          indx = files_def.init_getFiles();
        end
        case_info_set = false;
        while true
          if isempty(files_def.sock)
            [fldef, indx] = files_def.getFiles(indx);
          else
            [code_elmnt, ~, ~] = ...
              CodeElement.readElement(struct(), files_def.sock, inf);
            if ischar(code_elmnt)
              exc = MException('CSVidDecoder:doSimulationCase',...
                ['%s Error in CodeElement:readElement(): %s' prefix, code_elmnt]);
              throw(exc);
            elseif isscalar(code_elmnt) && isnumeric(code_elmnt) && ...
                code_elmnt == -1.
              fldef = [];
            elseif isa(code_elmnt, 'CSVidFile')
              if isempty(code_elmnt.name)
                continue;
              else
                fldef = files_def.getFilesByName(code_elmnt.name);
              end
            elseif isa(code_elmnt, 'CSVidCase')
              if isempty(code_elmnt.name)
                fldef = [];
              else
                error('Received CSVidCase with a non-empty name');
              end
            else
              error('Expected CSVidFile or CSVidCase, received %s',...
                class(code_elmnt));
            end
          end
          if isempty(fldef)
            break
          end
          
          if ~isempty(dec_opts) && isequal(dec_opts.init, -1)
            dec_opts.ref = fldef.input;
          end
          dec = CSVidDecoder(fldef, anls_opts, dec_opts);
          prc_opts = proc_opts;
          prc_opts.prefix = sprintf('%s%d]', prefix, k);
          dec.setPrefix(prc_opts.prefix);
          if isfield(proc_opts, 'blk_rng')
            dec.setBlkRange(proc_opts.blk_rng);
          end
          if iscell(fldef.enc_vid)
            if ~isempty(files_def.sock)
              enc_vid = files_def.sock;
            else
              enc_vid = fldef.enc_vid;
            end
          else
            enc_vid = fldef.enc_vid;
          end
          
          prof_ctrl(2, prc_opts.prof_spec);
                    
          result = dec.run(enc_vid, prc_opts);

          prof_ctrl(1, prc_opts.prof_spec);
                    
          if ~isempty(result)
            if ~isempty(result.psnr)
              psnr_str = sprintf('PSNR=%f4.1dB ', result.psnr);
            else
              psnr_str = '';
            end
            fprintf('%s file(%d) %s ==>\n  %s\n     %sDur. %6.2f sec.\n',...
              prc_opts.prefix, k, show_str(fldef.enc_vid), fldef.output, ...
              psnr_str, dec.total_dur);
            
            if ~proc_opts.no_stats
              cs_vid_io.add(result);
              
              if ~case_info_set
                cs_vid_io.setParams(struct(...
                  'msrmnt_input_ratio', dec.info.enc_opts.msrmnt_input_ratio,...
                  'qntzr_ampl_stddev', dec.info.enc_opts.qntzr_ampl_stddev,...
                  'qntzr_wdth_mltplr', dec.info.enc_opts.qntzr_wdth_mltplr));
                case_info_set = true;
              end
            end
          end
          k=k+1;
        end
        
        case_time = toc(case_start);
        fprintf('%s Case duration: %f\n', prefix, case_time);
      end
    end
    
    function simul_io_data = doSimulation(anls_opts, dec_opts, ...
        io_def, proc_opts)
      % Run decoding simulation on a multiple files in multiple conditions
      %   Input
      %     anls_opts - If missing or empty no measurements analysis is done.
      %             Otherwise it speciifies options for measurements
      %             analysis. can be an object of type CS_AnlsParams or
      %             something which can be used as an argument to construct
      %             such an object:
      %              A struct in which each field specify the property to
      %                 be changed.  An empty may be
      %                 used if no change to the defaults is necessary.
      %              A JSON string specifying such a struct.  If a field
      %                 value is a string beginning with an ampersand (&),
      %                 the field minus the ampersand prefix is evaluated
      %                 before assignment.
      %              A string containing an '<' followed by a file name. A
      %                JSON string is read from the file and converted to a
      %                struct as above.
      %     dec_opts - can be an object of type CS_DecParams or something which
      %            can be used as an argument to construct such an object:
      %              A struct in which each field specify the property to be
      %                 changed.  An empty struct or empty array may be used if
      %                 no change to the defaults is necessary.
      %              A JSON string specifying such a struct.  If a field value is
      %                 a string beginning with an ampersand (&), the field minus
      %                 the ampersand prefix is evaluated before assignment.
      %              A string containing an '<' followed by a file name. A JSON
      %                string is read from the file and converted to a struct as
      %                above.
      %     io_def   - specifies the base file names and directories.
      %     proc_opts (Optional) - a struct with any of the following
      %         fields which impact the processing mode (other fields are
      %         ignored):
      %         io_types - an optional cell array of the types to be
      %             processed (used as the optional argument fields in
      %             FilesDef constructor). This argument is ignored if
      %             io_def is a FilesDef object.
      %         output_id - Encoded input directory name.  Tnis is inserted
      %                 into output_dir in io_def.
      %         dec_id - Decoded output directory name. Inserted into dec_dir
      %                 in io_def. If it contains an asterisk the asterisk is
      %                 replaced by a time-stamp.
      %         prefix - (0ptional) an identifying prefix to add
      %                  before all printing. Default '<Nn>] '
      %         par_files - If true process files in parallel.
      %                     Default: false.
      %         par_cases - If non-zero, number of cases to process in
      %                     parallel. Default: 0
      %         blk_rng - If present specifies blocks range for
      %                   processing. A 2x3 array, upper row is minimum,
      %                   lower row is maximum
      %         use_gpu - If true, use GPU (default = false).
      %         use_single - If true, uss single precision (default = false)
      %         no_stats - If present and true, statistics are not computed,
      %         prof_spec - profiling specification. allowed values are:
      %                       0 - no profiling
      %                       1 - profile only run() function.
      %                       2 - profile everything.
      %                     default: 0
      
      simulation_start=tic;
      
      if nargin >= 4
        prof_ctrl(1, proc_opts);
      else
        prof_ctrl(1);
      end
    
      def_proc_opts = struct(...
        'output_id', '',...
        'dec_id', '*',...
        'prefix', '<Nn>] ',...
        'par_files', false,...
        'par_cases', 0,...
        'use_gpu', CompMode.defaultUseGpu(),...
        'use_single', false,...
        'no_stats', false,...
        'prof_spec', 0);
      
      if nargin >= 4
        flds = fieldnames(proc_opts);
        for k=1:length(flds)
          fld = flds{k};
          def_proc_opts.(fld) = proc_opts.(fld);
        end
      end
      
      proc_opts = def_proc_opts;

      proc_opts.dec_id = regexprep(proc_opts.dec_id, '*',...
        datestr(now,'yyyymmdd_HHMM'),'once');
      
      simul_io_data=SimulationInputOutputData();
      
      if ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
        anls_opts = CS_AnlsParams(anls_opts);
      end
      
      if ~isempty(dec_opts) && ~isa(dec_opts, 'CS_DecParams')
        dec_opts = CS_DecParams(dec_opts);
      end
      
      % Initialize input files
      if ~isfield(proc_opts, 'io_types')
        files_def = CSVidDecoder.set_output(io_def, ...,
          proc_opts.output_id, proc_opts.dec_id);
      else
        files_def = CSVidDecoder.set_output(io_def, ...
          proc_opts.output_id, proc_opts.dec_id, proc_opts.io_types);
      end
      
      if isempty(files_def.sock)
        par_cases = proc_opts.par_cases;
        case_dirs = files_def.getCaseDirs();
        n_cases = length(case_dirs);
        simul_data=cell(1,length(case_dirs));
      else
        par_cases = 0;
        simul_data=cell(1,1000); % arbitrary large size
      end
      
      %initialize storage to record how the simulation was run and the
      %results of the simulation
      fprintf('%s CSVidCodec.doSimulation() SVN version: %s\n%s  output dir=%s\n',...
        proc_opts.prefix, getSvnVersion(), ...
        proc_opts.prefix, files_def.getDecoderRoot());
      pprfx = proc_opts.prefix;
      if par_cases
        had_error = false(n_cases,1);
        case_opts = cell(1,n_cases);
        for j=1:n_cases
          case_opts{k} = proc_opts;
          case_opts{k}.prefix = sprintf('%s[%d] ', pprfx, k);
          case_opts{k}.case_dir = case_dirs{k};
          if case_opts.use_gpu
            case_opts{k}.gpus = ...
              proc_opts.gpus(1+mod(j,length(proc_opts.gpus)));
          end
        end
        for j = 1:par_cases:n_cases
          j_end = min(j+par_cases-1,n_cases);
          parfor k=j:j_end
            try
              simul_data{k} = ...
                CSVidDecoder.doSimulationCase(...
                anls_opts, dec_opts, files_def, case_opts{k});
              
              fprintf('%s Case %d of %d done\n', ...
                case_opts{k}.prefix, k, n_cases);
            catch exc
              had_error(k) = true;
              fprintf('%s **** Error: %s\n    %s\n', ...
                case_opts{k}.prefix, exc.identifier, exc.message);
              dbstack;
            end
          end
          if any(had_error)
            error('Error in parallel processing');
          end
          simul_io_data.setResults(simul_data(1:j_end));
          save([files_def.outputDir() '-simul_io_data.mat'],...
            '-mat');
        end
      else
        k=1;
        while true
          if isempty(files_def.sock)
            if k>n_cases
              break;
            end
            case_dir = case_dirs{k};
          else
            [code_elmnt, ~, ~] = ...
              CodeElement.readElement(struct(), files_def.sock, inf);
            if ischar(code_elmnt)
              exc = MException('CSVidDecoder:doSimulation',...
                ['%s Error in CodeElement:readElement(): ' pprfx, code_elmnt]);
              throw(exc);
            elseif isnumeric(code_elmnt) && isscalar(code_elmnt) && ...
                code_elmnt == -1
              % EOD case
              fprintf('doSimulation: EOD\n');
              break;
            elseif ~isa(code_elmnt, 'CSVidCase')
              error('Expected CSVidCase, received %s', class(code_elmnt));
            elseif isempty(code_elmnt.name)
              fprintf('doSimulation: Empty CSVidCase\n');
              break;
            end
            case_dir = code_elmnt.name;
          end
          
          case_opts = proc_opts;
          case_opts.prefix = sprintf('%s[%d]', pprfx, k);
          case_opts.case_dir = case_dir;
          simul_data{k} = ...
            CSVidDecoder.doSimulationCase(...
            anls_opts, dec_opts, files_def, case_opts);
          
          if isempty(files_def.sock)
            fprintf('%s Case %d of %d done\n', case_opts.prefix,...
              k, n_cases);
          else
            fprintf('%s Case %d done\n', case_opts.prefix, k);
          end
          simul_io_data.setResults(simul_data(1:k));
          save([files_def.outputDir() '-simul_io_data.mat'],...
            '-mat');
          k=k+1;
        end
        simul_data = simul_data(1:k-1); % limit length to actual used.
      end
      
      simul_io_data.setResults(simul_data);
      save([files_def.outputDir() 'simul_io_data.mat'],...
        '-mat');
      simul_dur=toc(simulation_start);
      
      fprintf('Simulation done in %f sec.\n', simul_dur);
      
    end
    
    % Run lossless coding conversion on multiple files in one condition
    %   Input
    %     conv_params - conv ersin parametres. a struct that may contain
    %         the following fields:
    %           lossless_coder
    %           lossless_coder_AC_gaus_thrsh
    %     files_def - A FilesDef object, or a JSON string defining such
    %           an object, or a string bgining with '<' which defines a
    %           file from which the JSON string can be read.
    %     params   - (optional) a struct which contains optioanl
    %                parameters. Its fields may be:
    %           prefix - prefix of printed lines
    %           case_dir - the case directory name (forces this name)
    %           io_params - A struct of the type used as a second
    %                       argumnt to the constructor of FilesDef
    %     proc_opts (Optional) - a struct with any of the following
    %         fields which impact the processing mode (other fields are
    %         ignored):
    %         io_types - (optional) cell array of types to be processed.
    %         output_id - (optional) If present, the output directory is
    %                     set to output_id.  If output_id contains an
    %                     sterisk, the sterisk is replaced by a
    %                     date.
    %         dec_id - Decoded output directory name. Inserted into dec_dir
    %                 in io_def. If it contains an asterisk the asterisk is
    %                 replaced by a time-stamp.
    %         io_types - an optional cell array of the types to be
    %             processed (used as the optional argument fields in
    %             FilesDef constructor). This argument is ignored if
    %             io_def is a FilesDef object.
    %         case_dir - the case directory name (forces this name)
    %         prefix - (0ptional) and identifying prefix to add
    %                  before all printing. Default: '] '
    %         par_files - If true process files in parallel.
    %                     Default: false.
    %         blk_rng - If present specifies blocks range for
    %                   processing. A 2x3 array, upper row is minimum,
    %                   lower row is maximum
    function cs_vid_io = doConvertLLC_Case(conv_params, files_def, ...
        proc_opts)
      case_start = tic;
      
      io_params = struct('identifier', struct());
      if nargin < 3
        proc_opts = struct();
      end
      if isfield(proc_opts, 'output_id')
        io_params.identifier.output = proc_opts.output_id;
      end
      if isfield(proc_opts, 'dec_id')
        io_params.identifier.dec = regexprep(proc_opts.dec_id, '*',...
          datestr(now,'yyyymmdd_HHMM'),'once');
      end
      if isfield(proc_opts, 'io_types')
        io_params.fields = proc_opts.io_types;
      end
      if ~isfield(proc_opts, 'prefix')
        proc_opts.prefix = ']';
      end
      prefix = proc_opts.prefix;
      if ~isfield(proc_opts, 'par_files')
        proc_opts.par_files = false;
      end
      
      if isa(files_def, 'FilesDef')
        files_def.specifyDirs(io_params.identifier);
      else
        files_def = FilesDef(files_def, io_params);
      end
      if isfield(proc_opts, 'case_dir')
        files_def.setDirs(struct('case', proc_opts.case_dir));
      end
      
      fprintf('%s Starting simulation case on: %s\n%s output directory=%s \n', ...
        prefix, files_def.caseDir(),  prefix, files_def.getDecoderDir());
      
      cs_vid_io = CSVideoCodecInputOutputData();
      cs_vid_io.clearResults();
      if proc_opts.par_files
        fdef_list = files_def.getAllFiles();
        params_fdef = cell(size(fdef_list));
        csv_inf = cell(size(fdef_list));
        for k=1:length(fdef_list)
          params_fdef{k} = conv_params;
        end
        parfor k=1:length(fdef_list)
          fldef = fdef_list(k);
          params_def = params_fdef{k};
          dec_prefix = sprintf('%s%d]', prefix, k);
          dec = CSVidDecoder(fldef, struct());
          dec.setPrefix(dec_prefix);
          if isfield(proc_opts, 'blk_rng')
            dec.setBlkRange(proc_opts.blk_rng);
          end
          csv_inf{k} = dec.convertLLC(dec_prefix,...
            params_def, fldef.mat, fldef.cnv_mat,...
            fldef.enc_vid, fldef.cnv_enc);
          fprintf('%s file(%d) %s converted\n     into %s\n',...
            prefix, k, fldef.enc_vid, fldef.cnv_enc);
        end
        for k=1:length(fdef_list)
          cs_vid_io.add(csv_inf{k});
        end
        if ~isempty(fdef_list)
          cs_vid_io.msrmnt_input_ratio = ...
            csv_inf{1}.msrmnt_input_ratio;
          cs_vid_io.qntzr_ampl_stddev = ...
            csv_inf{1}.qntzr_ampl_stddev;
          cs_vid_io.qntzr_wdth_mltplr = ...
            csv_inf{1}.qntzr_wdth_mltplr;
        end
      else
        indx = files_def.init_getFiles();
        k=1;
        while true
          [fldef, indx] = files_def.getFiles(indx);
          if isequal(fldef,[])
            if ~ isempty(dec)
              cs_vid_io.msrmnt_input_ratio =...
                csv_inf.msrmnt_input_ratio;
              cs_vid_io.qntzr_ampl_stddev = ...
                csv_inf.qntzr_ampl_stddev;
              cs_vid_io.qntzr_wdth_mltplr = ...
                csv_inf.qntzr_wdth_mltplr;
            end
            break
          end
          dec_prefix = sprintf('%s%d]', prefix, k);
          dec = CSVidDecoder(fldef, struct());
          dec.setPrefix(dec_prefix);
          if isfield(proc_opts, 'blk_rng')
            dec.setBlkRange(proc_opts.blk_rng);
          end
          csv_inf = dec.convertLLC(dec_prefix,...
            conv_params, fldef.mat, fldef.cnv_mat,...
            fldef.enc_vid, fldef.cnv_enc);
          cs_vid_io.add(csv_inf);
          k=k+1;
        end
        
        case_time = toc(case_start);
        fprintf('%s Case duration: %f\n', prefix, case_time);
      end
    end
    
    % Run lossless conversion on a multiple files in multiple conditions
    %   Input
    %     conv_params - conversion parametres. a struct that may contain
    %         the following fields:
    %           lossless_coder
    %           lossless_coder_AC_gaus_thrsh
    %     output_id - Encoded input directory name.  Tnis is inserted into
    %                output_dir in io_def.
    %     dec_id - Decoded output directory name. Inserted into dec_dir
    %              in io_def.
    %     io_def   - specifies the base file names and directories.
    %     proc_opts (Optional) - a struct with any of the following
    %     fields which impact the processing mode (other fields are
    %     ignored):
    %         io_types - an optional cell array of the types to be
    %             processed (used as the optional argument fields in
    %             FilesDef constructor). This argument is ignored if
    %             io_def is a FilesDef object.
    %         output_id - Encoded input directory name.  Tnis is inserted
    %                 into output_dir in io_def.
    %         dec_id - Decoded output directory name. Inserted into dec_dir
    %                 in io_def. If it contains an asterisk the asterisk is
    %                 replaced by a time-stamp.
    %         prefix - (0ptional) and identifying prefix to add
    %                  before all printing. Default '<Nn>] '
    %         par_files - If true process files in parallel.
    %                     Default: false.
    %         par_cases - If non-zero, number of cases to process in
    %                     parallel. Default: 0
    %         io_types - (optional) may be added to restrict the output types.
    %         blk_rng - If present specifies blocks range for
    %                   processing. A 2x3 array, upper row is minimum,
    %                   lower row is maximum
    function simul_io_data = doConvertLLC(conv_params, io_def, proc_opts)
      %start the stop watch
      simulation_start=tic;
      
      if nargin < 4
        proc_opts = struct();
      end
      
      if ~isfield(proc_opts, 'output_id')
        proc_opts.output_id = '';
      end
      proc_opts.output_id = regexprep(proc_opts.output_id, '*',...
        datestr(now,'yyyymmdd_HHMM'));
      
      if ~isfield(proc_opts, 'dec_id')
        proc_opts.dec_id = '*';
      end
      proc_opts.dec_id = regexprep(dec_id, '*',...
        datestr(now,'yyyymmdd_HHMM'),'once');
      
      if ~isfield(proc_opts, 'prefix')
        proc_opts.prefix = '';
      end
      
      if ~isfield(proc_opts, 'par_files')
        proc_opts.par_files = false;
      end
      
      if ~isfield(proc_opts, 'par_cases')
        proc_opts.par_cases = 0;
      end
      
      simul_io_data=SimulationInputOutputData();
      
      % Initialize input files
      if ~isfield(proc_opts, 'io_types')
        files_def = CSVidDecoder.set_output(io_def, ...,
          proc_opts.output_id, proc_opts.dec_id);
      else
        files_def = CSVidDecoder.set_output(io_def, ...
          proc_opts.output_id, proc_opts.dec_id, proc_opts.io_types);
      end
      
      case_dirs = files_def.getCaseDirs();
      n_cases = length(case_dirs);
      
      %initialize storage to record how the simulation was run and the
      %results of the simulation
      simul_data=cell(1,length(case_dirs));
      
      pprfx = proc_opts.prefix;
      if proc_opts.par_cases
        for j = 1:proc_opts.par_cases:n_cases
          j_end = min(j+proc_opts.par_cases-1,n_cases);
          parfor k=j:j_end
            case_opts = proc_opts;
            case_opts.prefix = sprintf('%s[%d] ', pprfx, k);
            case_opts.case_dir = case_dirs{k};
            simul_data{k} = ...
              CSVidDecoder.doConvertLLC_Case(conv_params,...
              files_def, case_opts);
            
            fprintf('%s Case %d of %d done\n', case_opts.prefix,...
              k, n_cases);
          end
          simul_io_data.setResults(simul_data(1:j_end));
          save([files_def.outputDir() '-simul_io_data.mat'],...
            '-mat');
        end
      else
        for k = 1:n_cases
          prefix = sprintf('%s[%d]', pprfx, k);
          case_dir = case_dirs{k};
          simul_data{k} = ...
            CSVidDecoder.doConvertLLC_Case(conv_params,...
            files_def, case_dir, ...
            struct('prefix', [prefix ']']));
          
          simul_io_data.setResults(simul_data(1:k));
          save([files_def.outputDir() '-simul_io_data.mat'],...
            '-mat');
          fprintf('%s Case %d of %d done\n', prefix, k, n_cases);
        end
      end
      
      simul_io_data.setResults(simul_data);
      save([files_def.outputDir() 'simul_io_data.mat'],...
        '-mat');
      simul_dur=toc(simulation_start);
      
      fprintf('Simulation done in %f sec.\n', simul_dur);
      
    end
     
  end
  
  methods (Static)
    function dec_data = calculateDecData(dec_data)
      start_time = tic;
      
      % Check if there is any output file which needs reconstruction
      if dec_data.reconstruction_needed || any(dec_data.slct_reconstruct)
        dec_info = dec_data.info;
        slv_opts = dec_data.solver_opts;
        dec_data = CSVidDecoder.unquantize(dec_data);
        
        % Do reconstruction
        [xvec, sideinfo] = ...
          CSVidDecoder.reconstructVideo(dec_data);
        if isfield(sideinfo, 'blk_psnr')
          dec_data.blk_psnr = sideinfo.blk_psnr;
        end
        
        dec_blks = dec_info.vid_region.pixelize(xvec, slv_opts.expand_level);
        if ~isempty(dec_data.ref_blks)
          rf_blks = vertcat(dec_data.ref_blks{:});
          dec_data.ref_blks = dec_info.vid_region.pixelize(rf_blks, ...
            slv_opts.expand_level);
        end
        
        if any(dec_info.enc_opts.blk_pre_diff)
          pre_diff_blks = dec_blks;
          dec_blks = dec_info.vid_region.do_multiDiffUnExtnd(...
            dec_blks, dec_info.enc_opts.blk_pre_diff);
        else
          pre_diff_blks = [];
        end
        
        if dec_data.dec_blks
          dec_data.dec_blks = dec_blks;
        else
          dec_data.dec_blks = [];
        end
        
        % Create pre_diff_blks if necessary
        if dec_data.pre_diff_blks && any(dec_info.enc_opts.blk_pre_diff)
          dec_data.pre_diff_blks = pre_diff_blks;
        else
          dec_data.pre_diff_blks = [];
        end
        
      else
        dec_blks = [];
        dec_data.dec_blks = [];
        dec_data.pre_diff_blks = [];
        dec_data.dec_mrk_blks = [];
      end
      
      if ~dec_data.solver_opts.whole_frames
        dec_data = CSVidDecoder.markDecData(dec_data, dec_blks);
      end
      
      dec_data.proc_dur = dec_data.proc_dur + toc(start_time);
    end
    
    function m_dec_data = jointCalculateDecData(m_dec_data)
      start_total = tic;
      
      n_used = 0;
      u_indx = zeros(size(m_dec_data));
      u_mtrx = cell(size(m_dec_data));
      u_sns_xpnd_U = cell(size(m_dec_data));
      u_sns_xpnd_R = cell(size(m_dec_data));
      u_sparser_mtx = cell(size(m_dec_data));
      u_sparser_xpnd = cell(size(m_dec_data));
      u_sgnl_len = zeros(size(m_dec_data));
      u_msrs = cell(size(m_dec_data));
      u_blk_indx = cell(size(m_dec_data));
      u_ref_blks = cell(size(m_dec_data));
      frm_data = [];
      
      for k=1:length(m_dec_data)
        start_time = tic;
        dec_data = m_dec_data{k};
        
        if dec_data.reconstruction_needed
          if n_used == 0
            frm_data = dec_data;
          end
          n_used = n_used+1;
          u_indx(n_used) = k;
          dec_data = CSVidDecoder.unquantize(dec_data);
          m_dec_data{k}.dc_val = dec_data.dc_val;
          
          u_mtrx{n_used} = dec_data.info.sens_mtrx;
          u_sns_xpnd_U{n_used} = dec_data.sparser_info.sns_xpnd.U;
          u_sns_xpnd_R{n_used} = dec_data.sparser_info.sns_xpnd.R;
          sprsr_args = dec_data.sparser_info.sprsr.getArgs();
          u_sparser_mtx{n_used} = sprsr_args.sprs_mtx;
          u_sparser_xpnd{n_used} = sprsr_args.expndr;
          u_sgnl_len(n_used) = u_sns_xpnd_U{n_used}.nCols();
          u_msrs{n_used} = dec_data.cs_msrs;
          u_blk_indx{n_used} = dec_data.info.vid_region.blk_indx;
          u_ref_blks{n_used} = dec_data.ref_blks;
        else
          dec_data.dec_blks = [];
          dec_data.pre_diff_blks = [];
          dec_data.dec_mrk_blks = [];
        end
        
        m_dec_data{k}.proc_dur = m_dec_data{k}.proc_dur + toc(start_time);
      end
      
      ttl_dur = toc(start_total);
      
      if n_used
        start_time = tic;
        start_total = tic;
        
        u_sgnl_len = u_sgnl_len(1:n_used);
        u_sgnl_end = cumsum(u_sgnl_len(:));
        u_sgnl_bgn = 1 + [0; u_sgnl_end(1:end-1)];
        frm_data.info.sens_mtrx = SensingMatrixBlkDiag.constructBlkDiag(...
          u_mtrx(1:n_used));
        vrgn = frm_data.info.vid_region;
        frm_data.info.vid_region = VidRegion(...,
          vertcat(u_blk_indx{1:n_used}), vrgn.blkr, vrgn.zext, vrgn.wext);
        sprsr_args = frm_data.sparser_info.sprsr.getArgs();
        sprsr_args.vdrg = frm_data.info.vid_region;
        sprsr_args.expndr =...
          SensingMatrixBlkDiag.constructBlkDiag(u_sparser_xpnd(1:n_used));
        sprsr_args.sprs_mtx = ...
          SensingMatrixBlkDiag.constructBlkDiag(u_sparser_mtx(1:n_used));
        frm_data.sparser_info.sprsr = BaseSparser(sprsr_args);
        frm_data.sparser_info.sns_xpnd = struct(...
          'U', SensingMatrixBlkDiag.constructBlkDiag(u_sns_xpnd_U(1:n_used)),...
          'R', SensingMatrixBlkDiag.constructBlkDiag(u_sns_xpnd_R(1:n_used)));
        frm_data.sparser_info.sprsr.use_gpu = m_dec_data{1}.proc_opts.use_gpu;
        frm_data.sparser_info.sprsr.use_single = m_dec_data{1}.proc_opts.use_single;
        frm_data.cs_msrs = vertcat(u_msrs{1:n_used});
        frm_data.ref_blks = vertcat(u_ref_blks{1:n_used});
        frm_data.prefix = sprintf('%s blks of frms %d:%d]', ...
          frm_data.prfx, min(vrgn.blk_indx(:,3)), max(vrgn.blk_indx(:,3)));
        pxlmx = frm_data.info.raw_vid.getPixelMax();
        
        [xvec, sideinfo] = CSVidDecoder.reconstructVideo(frm_data);
        
        frm_dur = toc(start_time);
        
        for k=1:n_used
          start_time = tic;
          dec_data = m_dec_data{u_indx(k)};
          dec_info = dec_data.info;
          slv_opts = dec_data.solver_opts;
          blk_xvec = xvec(u_sgnl_bgn(k):u_sgnl_end(k));
          
          dec_blks = dec_info.vid_region.pixelize(blk_xvec, slv_opts.expand_level);
          if ~isempty(dec_data.ref_blks)
            rf_blks = vertcat(dec_data.ref_blks{:});
            dec_data.ref_blks = dec_info.vid_region.pixelize(rf_blks, ...
              slv_opts.expand_level);
          end
          
          if any(dec_info.enc_opts.blk_pre_diff)
            pre_diff_blks = dec_blks;
            dec_blks = dec_info.vid_region.do_multiDiffUnExtnd(...
              dec_blks, dec_info.enc_opts.blk_pre_diff);
          else
            pre_diff_blks = [];
          end
          
          if dec_data.dec_blks
            dec_data.dec_blks = dec_blks;
          else
            dec_data.dec_blks = [];
          end
          
          % Create pre_diff_blks if necessary
          if dec_data.pre_diff_blks
            dec_data.pre_diff_blks = pre_diff_blks;
          else
            dec_data.pre_diff_blks = [];
          end
          
          if isfield(sideinfo, 'blk_psnr')
            blk_ref = dec_data.ref_blks;
            if any(dec_info.enc_opts.blk_pre_diff)
              blk_ref = dec_info.vid_region.multiDiffExtnd(...
                blk_ref, dec_info.enc_opts.blk_pre_diff, slv_opts.expand_level);
            end
            blk_ref = dec_info.vid_region.vectorize(blk_ref);
            dec_data.blk_psnr = SqrErr.compPSNR(blk_ref, blk_xvec, pxlmx);
          end
          
          dec_data = CSVidDecoder.markDecData(dec_data, dec_blks);
          
          % Prevent reports per block unless report_blk=2.
          if dec_data.proc_opts.report_blk > 0
            dec_data.proc_opts.report_blk = ...
              dec_data.proc_opts.report_blk - 1;
          end
          
          m_dec_data{k} = dec_data;
          
          % Split the joint processing time amont all blocks
          m_dec_data{k}.proc_dur = m_dec_data{k}.proc_dur + ...
            frm_dur/n_used + toc(start_time);
          
        end
        
        ttl_dur = ttl_dur + toc(start_total);
        
        if frm_data.proc_opts.report_frm
          if isfield(sideinfo, 'blk_psnr')
            psnr_str = sprintf(' PSNR=%4.1f', sideinfo.blk_psnr);
          else
            psnr_str = '';
          end
          fprintf('%s Proc Dur=%6.2f. sec.%s\n', ...
            frm_data.prefix, ttl_dur, psnr_str);
        end
        
      end
      
    end
    
    function dec_data = analyzeDecData(dec_data)
      start_time = tic;
      
      dec_data.cs_msrs = [];
      
      if dec_data.analysis_needed
        dec_data = CSVidDecoder.unquantize(dec_data);
        vid_region = dec_data.vid_region;
        
        for k=1:size(vid_region.blk_indx,1)
          vdrg =  VidRegion(vid_region.blk_indx(k,:), vid_region.blkr,...
            vid_region.zext, vid_region.wext);
          [~, dec_data.blk_motion{k}] =...
            next_msrs_xcor(dec_data.cs_msrs, dec_data.blk_sens_mtrcs{k}, ...
            vdrg, dec_data.anls_opts);
          dec_data.has_motion(k) = ~isempty(dec_data.blk_motion{k}) && ...
            dec_data.blk_motion{k}.motionFound();
          
          % If necessary ignore motion if it is on edge block
          dec_data.has_motion(k) = dec_data.has_motion(k) && ...
            ~(dec_data.anls_opts.ignore_edge && ...
            (any(vdrg.blk_indx(1,1:2)==[1 1]) ||...
            any(vdrg.blk_indx(1:2)==vdrg.blkr.blk_cnt(1:2))));
          
          if dec_data.has_motion(k)
            fprintf('%s %s Motion: %s\n', dec_data.prefix,...
              show_str(dec_data.info.vid_region.blk_indx(k,:)),...
              dec_data.blk_motion{k}.report());
            
            if dec_data.dec_slct_mrk_blks;
              dec_data.slct_reconstruct = true;
            end
          end
               
        end
      end
      dec_data.reconstruction_needed = dec_data.reconstruction_needed || ...
        all(dec_data.slct_reconstruct);
      
      dec_data.proc_dur = dec_data.proc_dur + toc(start_time);
    end
    
    function dec_data = unquantize(dec_data)
      if ~isempty(dec_data.cs_msrs)
        return;
      end
      
      [cs_msrs, clipped_indices] = ...
        dec_data.info.quantizer.unquantize(dec_data.q_msrs);
      cs_msrs = ...
        dec_data.info.sens_mtrx.unsortNoClip(cs_msrs);
      dec_data.dc_val = dec_data.info.sens_mtrx.getDC(cs_msrs);
      dec_data.info.sens_mtrx.setZeroedRows(clipped_indices);
      dec_data.cs_msrs = cs_msrs;
    end
    
    function dec_data = markDecData(dec_data, dec_blks)
      pxmx = dec_data.info.raw_vid.getPixelMax();
      
      slct_blks = [];
      
      if dec_data.dec_mrk_blks
        dec_data.dec_mrk_blks = mark_blk_boundaries(dec_blks,...
          dec_data.info.vid_blocker.ovrlp, dec_data.info.enc_opts.conv_rng,...
          0, pxmx);
        dec_data.dec_mrk_blks = dec_data.info.vid_region.drawMotionMarker(...
          dec_data.dec_mrk_blks, dec_data.solver_opts.expand_level, ...
          [0.5,0.5], [0, pxmx], dec_data.blk_motion{1});
      else
        dec_data.dec_mrk_blks = [];
      end
      
      % Create the selected blocks slct_mrk_blks, if motion
      % happened
      if any(dec_data.slct_reconstruct & dec_data.has_motion)
        % <<<< Fix this call
        slct_blks = dec_data.info.vid_region.drawMotionMarker(...
          dec_blks, [0.5,0.5], dec_data.solver_opts.expand_level,...
          [0, pxmx], dec_data.blk_motion{1});
        
      end
      
      % Create the selected blocks slct_mrk_blks,
      if dec_data.dec_slct_mrk_blks
        if isempty(slct_blks)
          slct_blks = dec_data.info.vid_region.getEmpty();
          slct_blks{1} = (pxmx/2)*ones(size(slct_blks{1}));
          if ~isempty(dec_data.blk_motion) && ...
              dec_data.blk_motion{1}.motionFound() > 0
            slct_blks = dec_data.info.vid_region.drawMotionMarker(...
              slct_blks, dec_data.solver_opts.expand_level, [0.5,0.5], ...
              [0, pxmx], dec_data.blk_motion{1});
          end
        end
        dec_data.dec_slct_mrk_blks = slct_blks;
      else
        dec_data.dec_slct_mrk_blks = [];
      end
      
      % Create ref_mrk_blks
      if dec_data.ref_mrk_blks
        dec_data.ref_mrk_blks = mark_blk_boundaries(dec_data.ref_blks,...
          dec_data.info.vid_blocker.ovrlp, dec_data.info.enc_opts.conv_rng,...
          0, pxmx);
        dec_data.ref_mrk_blks = dec_data.info.vid_region.drawMotionMarker(...
          dec_data.ref_mrk_blks, dec_data.solver_opts.expand_level,...
          [0.5,0.5], [0, pxmx], dec_data.blk_motion{1});
      else
        dec_data.ref_mrk_blks = [];
      end
    end
    
    % recover the original signal from compressive sensing measurements
    function [xvec, sideinfo] = reconstructVideo(dec_data)
      
      dec_info = dec_data.info;
      slv_opts = dec_data.solver_opts;
      ref_blks = dec_data.ref_blks;
      prefix = dec_data.prefix;
      cs_msrs = dec_data.cs_msrs;
      
      sparser = dec_data.sparser_info.sprsr;
      sens_expndr = dec_data.sparser_info.sns_xpnd;
      sens_mtrx = SensingMatrixCascade.constructCascade(...
        {dec_info.sens_mtrx, sens_expndr.U});
      cnstrnts = dec_info.vid_region.compConstraints(slv_opts.expand_level);
      sens_mtrx = SensingMatrixConcat.constructConcat({sens_mtrx,...
        SensingMatrixCascade.construct({SensingMatrixScaler(cnstrnts.nRows(),...
        (sens_mtrx.norm()/cnstrnts.norm())), cnstrnts})});
      cs_msrs = [cs_msrs; zeros(cnstrnts.nRows(),1)];
      sideinfo = struct();
      
      if slv_opts.use_old
        % Use old method
        blks = CSVidDecoder.recoverInputFromCSMeasurements( ...
          sens_mtrx, sparser, cs_msrs, slv_opts.use_old);
        xvec = dec_info.vid_region.vectorizer(blks);
        in
      else
        % Use reference
        cmp_blk_psnr = [];
        if slv_opts.init == -1 || slv_opts.cmpr_blk_by_blk
          % Check for initial guess solution
          init_reg = ref_blks;
          if isempty(init_reg)
            if slv_opts.init == -1
              slv_opts.init = 0;
            end
          else
            if any(dec_info.enc_opts.blk_pre_diff)
              init_reg = dec_info.vid_region.multiDiffExtnd(...
                init_reg, dec_info.enc_opts.blk_pre_diff, slv_opts.expand_level);
            end
            init_reg = dec_info.vid_region.vectorize(init_reg);
            if slv_opts.init == -1
              slv_opts.init = init_reg;
            end
            
            if slv_opts.cmpr_blk_by_blk
              pxlmax = dec_info.raw_vid.getPixelMax();
              cmp_blk_psnr = @(blk) SqrErr.compPSNR(init_reg, blk, pxlmax);
            end
          end
        end
        
        b_idx = dec_data.info.vid_region.blk_indx;
        q_step = dec_data.info.quantizer.q_params(b_idx(1), b_idx(2), 1).intvl;
        pix_stdv_err = ...
          dec_data.info.quantizer.q_params(b_idx(1), b_idx(2), 1).msrs_noise;
        proc_params = struct('prefix', prefix, 'cmp_blk_psnr', cmp_blk_psnr);
        if slv_opts.Q_msrmnts && q_step ~= 0
          if nargout >= 2
            [xvec, blk_done, ...
              sideinfo.lambda, sideinfo.beta, sideinfo.out] = ...
              solveQuant(sens_mtrx, cs_msrs,...
              sparser, slv_opts, q_step, pix_stdv_err, proc_params);
          else
            [xvec, blk_done] = ...
              solveQuant(sens_mtrx, cs_msrs,...
              sparser, slv_opts, q_step, pix_stdv_err, proc_params);
          end
        else
          if nargout >= 2
            [xvec, blk_done, ...
              sideinfo.lambda, sideinfo.beta, sideinfo.out] = ...
              solveExact(sens_mtrx, cs_msrs,...
              sparser, slv_opts, q_step, pix_stdv_err, proc_params);
          else
            [xvec, blk_done] = ...
              solveExact(sens_mtrx, cs_msrs,...
              sparser, slv_opts, q_step, pix_stdv_err, proc_params);
          end
        end
        if ~blk_done
          fprintf('%s ---- region %s did not converge!\n', prefix,...
            show_str(dec_info.vid_region.blk_indx));
        end
        
        xvec = sens_expndr.R.multVec(xvec);
        if ~isempty(cmp_blk_psnr)
          sideinfo.blk_psnr = cmp_blk_psnr(xvec);
        end
      end
    end
  end
  
  methods (Static, Access=protected)
    
    function files_def  = set_output(io_def, output_id, dec_id, io_types)
      io_id = struct();
      if nargin >= 2
        io_id.output = output_id;
        if nargin >= 3
          io_id.dec = dec_id;
        end
      end
      
      % Initialize input files
      if isa(io_def,'FilesDef')
        files_def = io_def;
        files_def.specifyDirs(io_id);
      else
        io_params = struct('identifier', io_id);
        if nargin >= 4
          io_params.fields = io_types;
        end
        files_def = FilesDef(io_def, io_params);
      end
      files_def.makeOutputDir();
    end
  end
  
end

