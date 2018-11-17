classdef CSVidCodec < handle
  % This object contains methods to do video encoding and decoding with compressive
  % sensing.  The codec performs
  % the following steps:
  % 3) for each block,
  %   --> converts each color pixel (Y,U,V) of block into only luminance components
  %   --> takes compressed sensed measurements
  %   --> quantizes measurements
  %   --> encodes measurements with arithmetic coding
  % 1) reads in raw color video
  % 2) breaks the video into blocks
  %   --> decodes measurements with arithmetic coding
  %   --> recovers input block using TVAL3
  % 4) deblocks video (e.g. put video back together from block)
  %
  % It also contains methods to do an automated series of tests. See
  % "doSimulation" method for details.
  %
  % To plot the results of the simulation see the method
  %  plotSimulationInputOutputData
  %
  % Yen-ming Mark Lai (summer intern at Bell Labs, Murray Hill, NJ)
  % University of Maryland, College Park
  % ylai@amsc.umd.edu
  %
  % 8/12/2011
  
  properties (Constant=true)
    
    debug=0; %1 to turn on debugging (verbose output), 0 to turn off
    
    %         report_msrmnt_xcor = false;
    report_msrmnt_xcor = true;
    
  end
  
  properties
    proc_opts= struct(...,
      'prefix', ']', ...
      'report_blk', true,...
      'report_frm', true,...
      'inp_anls', 1,...
      'keep_sock', false,...
      'use_gpu', CompMode.defaultUseGpu(),...
      'use_single', false,...
      'no_stats', false,...
      'check_coding', 0);
    
    enc_opts;
    anls_opts;
    dec_opts;
    
    raw_vid_in % VidBlocksIn object for reading input blocks
    enc_out = {};  % Cell array of CodeDest objects for measurements output
    n_raw_pxls; % numbers of raw pixels in a block indexed by spatial postion.
    n_msrs_blk; % numbers of measurements per block indexed by spatial postion.
    inp_copy;  % VidBlocksOut for writing a copy of the input (for debugging)
    dec_inp;  % CodePipeArray interface to decoder.
    
    n_parblk = 0;
    n_parfrm = 0;
    
    % If parfrm != 0. beginnings and ends of samples of each block in
    % larger xvec vector
    blk_bgn = [];
    blk_end = [];
    blk_z_indx = [];  % indices in frame blocks, with temporal index starting at 0
    
    sns_mtrx_mgr;
    
    %max number of measurements in a block
    n_msrs_max = 0;
    
    % Pre-diff output files
    enc_pre_diff_blks;
    tst_pre_diff_blks;
    
    % Analysis outputs
    inp_mark_blks;
    inp_vmark_blks;
    enc_mark_blks;
    inp_anls;
    inp_sav;
    inp_vanls;
    inp_vsav;
    enc_anls;
    enc_sav;
    
    chk_bgrnd;
    
    % Statistics per block
    blk_data_list
    
    % timers
    file_start_time;
  end
  
  properties(Access = private)
    % Cleanup items
    mex_context_clnp = [];
  end
  
  methods
    function obj = CSVidCodec(enc_opts, anls_opts, dec_opts, proc_opts)
      % Constructor of CS Codec
      % Arguments:
      % enc_opts - a CS_EncVidParams objects or something which can be used
      %            as an argument to construct such an object:
      %              A struct in which each field specify the property to be
      %                 changed.  An empty struct or empty array may be used if
      %                 no change to the defaults is necessary.
      %              A JSON string specifying such a struct.  If a field value is
      %                 a string beginning with an ampersand (&), the field minus
      %                 the ampersand prefix is evaluated before assignment.
      %              A string containing an '<' followed by a file name. A JSON
      %                string is read from the file and converted to a struct as
      %                above.
      % anls_opts - If missing or empty no measurements analysis is done.
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
      %
      % dec_opts -  if present and not empty perform decoding. dec_opts
      %            can be an object of type CS_DecParams or something which
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
      % proc_opts - (optional) a struct with any of the following
      %              fields which impact the processing mode (other fields
      %              are ignored):
      %           prefix - prefix to prepend to messagees
      %           report_blk - report completion of each block
      %                        (default: true).
      %           report_frm - Report when frms are completed (default:
      %                        true)
      %           inp_anls - mode of input file analysis (default=1):
      %                        0 - never
      %                        1 - only if ~do_encode
      %                        2 - always
      %           keep_sock - For the case that the input is a socket: If
      %                      defined and true, the socket is not closed upon
      %                      exit. Otherwise it is closed.
      %           use_gpu - If true, use GPU (default = false).
      %            use_single - If true, uss single precision (default = false)
      %           no_stats - If present and true, statistics are not computed,
      %           check_coding - relevant only when decoding is done. If 0
      %                        (defaut) code elements are lossless encoded for
      %                        computing length, but passed to the decoder as
      %                        is, so the decoder does not have to do the
      %                        lossless decoding. If 1, the losslessly
      %                        encoded code elemens are passed to the decoder
      %                        and losslessly decoded. 2 is the same as 1 and
      %                        in addition, the code elements decoded by the
      %                        decoder are compared to the original ones at
      %                        the encoder and anf error is thrown if there is
      %                        a difference.

      obj.mex_context_clnp = initMexContext();
      
      if nargin < 4
        proc_opts = struct();
        if nargin < 3
          dec_opts = [];
          if nargin < 2
            anls_opts = [];
          end
        end
      end
      
      flds = fieldnames(proc_opts);
      for k=1:length(flds);
        fld = flds{k};
        obj.proc_opts.(fld) = proc_opts.(fld);
      end
      if obj.proc_opts.use_gpu
        if ~isfield(obj.proc_opts, 'gpus')
          obj.proc_opts.gpus = find_gpus();
        end
        if isempty(obj.proc_opts.gpus)
          obj.proc_opts.use_gpu = false;
        else
          init_gpu(obj.proc_opts.gpus(1));
        end
      end
      if ~obj.proc_opts.use_gpu && CompMode.defaultUseGpu()
        error('Specified no_gpu while MEX SW is compiled for GPU');
      end
      
      CompMode.setDefaultUseSingle(obj.proc_opts.use_single);
      
      if isa(enc_opts, 'CS_EncVidParams')
        obj.enc_opts = enc_opts;
      else
        obj.enc_opts = CS_EncVidParams(enc_opts);
      end
      obj.enc_opts.use_single = obj. proc_opts.use_single;
      obj.enc_opts.use_gpu = obj. proc_opts.use_gpu;
      
      if ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
        obj.anls_opts = CS_AnlsParams(anls_opts);
      else
        obj.anls_opts = anls_opts;
      end
      
      if ~isempty(dec_opts) && ~isa(dec_opts,'CS_DecParams')
        obj.dec_opts = CS_DecParams(dec_opts);
      else
        obj.dec_opts = dec_opts;
      end
    end
    
    function delete(obj)
      % Close output socket if necessary
      if ~isfield(obj.proc_opts, 'keep_sock') || ~obj.proc_opts.keep_sock
        CodeDest.deleteCodeDest(obj.enc_out);
      end
    end
    
    function [enc_info, decoder] = init(obj, fdef)
      % Initializes the coded for running with the given I/O set
      %   Input:
      %     obj - CS this codec object
      %     fdef - A struct as produced by FilesDef.getFiles() defining I/O files.
      %   Output:
      %     enc_info - a struct cotaining state information of the encoder
      %     dec_info - a struct cotaining state information of the decoder (empty
      %                if decoding not done)
      
      obj.file_start_time = tic;
      fprintf(1,'%s Version %s. Processing %s\n', obj.proc_opts.prefix, ...
        getSvnVersion(), fdef.input);
      
      enc_info = struct(...
        'enc_opts', obj.enc_opts,...
        'proc_opts', obj.proc_opts, ...
        'anls_opts', obj.anls_opts,...
        'dec_opts', obj.dec_opts,...
        'do_encode', ~isempty(obj.enc_opts) &&...
        ~strcmp(obj.enc_opts.msrmnt_mtrx.type,'NONE'),...
        'blk_indx', [],...
        'sens_mtrx', []...
        );
      
      % Initialize reading in of raw video
      vid_in_params = struct();
      if ~isempty(obj.dec_opts)
        vid_in_params.w_type_d = obj.dec_opts.wnd_type;
      end
      
      obj.raw_vid_in = VidBlocksIn(vid_in_params, fdef.input, VidBlocker.BLK_STT_ZEXT,...
        obj.enc_opts);
      
      if enc_info.enc_opts.n_frames == -1 || ...
          enc_info.enc_opts.n_frames  > obj.raw_vid_in.vid_size(1,3)
        enc_info.enc_opts.setParams(struct('n_frames',...
          obj.raw_vid_in.vid_size(1,3)));
      end
      
      enc_info.vid_blocker = VidBlocker(obj.raw_vid_in);
      enc_info.vid_blocker.use_gpu = enc_info.proc_opts.use_gpu;
      enc_info.vid_blocker.use_single = enc_info.proc_opts.use_single;
      enc_info.raw_vid = enc_info.vid_blocker.vid_info;
      
      obj.sns_mtrx_mgr = SensingMatrixManager(obj.enc_opts, enc_info.vid_blocker);
      
      blk_cnt = enc_info.vid_blocker.blk_cnt;
      obj.n_raw_pxls = obj.raw_vid_in.cntOrigRawPxls();
      
      if isfield(fdef, 'inp_copy')
        obj.inp_copy = VidBlocksOut(enc_info.vid_blocker, fdef.inp_copy, false, ...
          VidBlocker.BLK_STT_ZEXT);
        obj.inp_copy.use_gpu = enc_info.proc_opts.use_gpu;
        obj.inp_copy.use_single = enc_info.proc_opts.use_single;
      else
        obj.inp_copy = [];
      end
      
      enc_info.fdef = fdef;
      
      obj.n_parfrm = obj.enc_opts.par_blks;
      obj.n_parblk = obj.n_parfrm * blk_cnt(1) * blk_cnt(2);
      if obj.n_parfrm > 0
        obj.raw_vid_in.setBlkFrm(obj.n_parfrm);
        obj.blk_end = ...
          enc_info.vid_blocker.blkSizes(VidBlocker.BLK_STT_ZEXT) * ...
          ones(1,obj.n_parfrm);
        obj.blk_end = cumsum(obj.blk_end(:));
        obj.blk_bgn = [1; (obj.blk_end(1:end-1)+1)];
        
        obj.blk_z_indx = zeros(obj.n_parblk,3);
        b = 0;
        for t = 0:(obj.n_parfrm-1)
          for h = 1:blk_cnt(2)
            for v = 1:blk_cnt(1)
              b = b + 1;
              obj.blk_z_indx(b,:) = [v, h, t];
            end
          end
        end
      end
      
      enc_info.do_inp_anls = ~isempty(obj.anls_opts) &&...
        (enc_info.proc_opts.inp_anls==2 || ...
        (enc_info.proc_opts.inp_anls==1 && ~enc_info.do_encode)) &&...
        (isfield(enc_info.fdef, 'inp_anls') ||...
        isfield(enc_info.fdef, 'inp_sav') || ...
        isfield(enc_info.fdef, 'inp_mark') );
    
    if enc_info.do_encode
      if ~isempty(enc_info.dec_opts) && ~(isfield(enc_info.fdef, 'output') || ...
          isfield(enc_info.fdef, 'dec_mark') || isfield(enc_info.fdef, 'dec_slct_mark') ||...
          isfield(enc_info.fdef, 'dec_anls') || isfield(enc_info.fdef, 'dec_sav') || ...
          isfield(enc_info.fdef, 'dec_pre_diff') || ...
          isfield(enc_info.fdef, 'tst_pre_diff'))
        enc_info.dec_opts = [];
      elseif isempty(enc_info.dec_opts) && (~isempty(obj.anls_opts) && (...
          isfield(enc_info.fdef,'dec_anls') || isfield(enc_info.fdef,'dec_sav')))
        enc_info.dec_opts = CS_DecParams();
        flds = fields(enc_info.fdef);
        for k = 1:length(flds);
          fld = flds{k};
          if any(strcmp(fld, {'output', 'dec_mark', 'dec_slct_mark',...
              'dec_anls', 'dec_sav', ...
              'dec_pre_diff', 'tst_pre_diff'}))
            enc_info.fdef = rmfield(enc_info.fdef, fld);
          end
        end
      end
      
      if ~isempty(enc_info.anls_opts) && isfield(enc_info.fdef, 'inp_vanls')
        enc_info.anls_opts.setParams(struct('chk_ofsts', true));
      end
      
      if isfield(enc_info.fdef, 'enc_vid')
        obj.enc_out = CodeDest.constructCodeDest(enc_info.fdef.enc_vid);
        if ~iscell(obj.enc_out)
          obj.enc_out = {obj.enc_out};
        end
      end
      
      obj.n_msrs_blk = ceil(obj.n_raw_pxls * obj.enc_opts.msrmnt_input_ratio);
      
      if obj.enc_opts.random.rpt_temporal && ...
          ~isempty(enc_info.anls_opts) && enc_info.anls_opts.chk_bgrnd.mx_avg
        obj.chk_bgrnd = cell(blk_cnt(1),blk_cnt(2));
        for k=1:numel(enc_info.chk_bgrnd)
          enc_info.chk_bgrnd{k} = BgrndMsrs(obj.enc_opts.random.rpt_temporal,...
            enc_info.anls_opts.chk_bgrnd.mx_avg, ...
            enc_info.anls_opts.chk_bgrnd.mn_dcd,...
            enc_info.anls_opts.chk_bgrnd.thrsh);
        end
      else
        obj.chk_bgrnd = [];
      end
      
      if ~isempty(enc_info.dec_opts)
        obj.dec_inp = CodePipeArray(4096, 8*4096);
        obj.enc_out = [{obj.dec_inp} obj.enc_out];
        
        if enc_info.dec_opts.init == 2
          enc_info.dec_opts.ref = enc_info.fdef.input;
        end
        
        dec_proc_opts = enc_info.proc_opts;
        dec_proc_opts.report_blk = false;
        dec_proc_opts.report_frm = false;
        decoder = CSVidDecoder(enc_info.fdef, enc_info.anls_opts, enc_info.dec_opts, ...
          dec_proc_opts);
      else
        if enc_info.proc_opts.check_coding
          fprintf('%s No check_coding while dec_opts is []!\n', ...
            obj.proc_opts.prefix);
        end
        decoder = [];
      end
      
      if length(obj.enc_out) == 1
        obj.enc_out = obj.enc_out{1};
      end
      
      obj.open_analysis_output_files(enc_info);
    
      if any(enc_info.enc_opts.blk_pre_diff)
        if isfield(enc_info.fdef, 'enc_pre_diff')
          obj.enc_pre_diff_blks = ...
            VidBlocksOut(enc_info.vid_blocker, enc_info.fdef.enc_pre_diff, true,...
            VidBlocker.BLK_STT_ZEXT);
        else
          obj.enc_pre_diff_blks = ...
            VidBlocksOut(enc_info.vid_blocker, [], true, ...
            VidBlocker.BLK_STT_ZEXT);
        end
        obj.enc_pre_diff_blks.use_gpu = enc_info.proc_opts.use_gpu;
        obj.enc_pre_diff_blks.use_single = enc_info.proc_opts.use_single;
        
        if isfield(enc_info.fdef, 'tst_pre_diff')
          obj.tst_pre_diff_blks = ...
            VidBlocksOut(enc_info.vid_blocker, enc_info.fdef.tst_pre_diff, true,...
            VidBlocker.BLK_STT_ZEXT);
          enc_info.tst_pre_diff = true;
        else
          % Unnecessary if not written out
          enc_info.tst_pre_diff = false;
        end
        obj.tst_pre_diff_blks.use_gpu = enc_info.proc_opts.use_gpu;
        obj.tst_pre_diff_blks.use_single = enc_info.proc_opts.use_single;
        
        %       if ~isempty(decoder)
        %         decoder.initRefPreDiff(enc_info.raw_vid.getPixelMax(),...
        %           enc_info.raw_vid.createEmptyVideo(0));
        %       end
      end
            
      % Create a uniform quantizer
      enc_info.quantizer = UniformQuantizer.makeUniformQuantizer(obj.enc_opts,...
        enc_info.vid_blocker, obj.sns_mtrx_mgr, enc_info.proc_opts.use_gpu);
      
    else  % ~ enc_info.do_encode
      
      enc_info.anls_opts = [];
      enc_info.dec_opts = [];
      
    end
    
    if ~enc_info.proc_opts.no_stats
      %initialize storage for data about each block processed by
      %this codec
      obj.blk_data_list=cell(prod(blk_cnt),1);
    end
    
   fprintf('%s %s %dX%d, %d frms, %f fps, %s blocks, initialized in %.1f sec.\n', ...
     obj.proc_opts.prefix, ...
      enc_info.raw_vid.type, enc_info.raw_vid.height, enc_info.raw_vid.width, ...
      enc_info.raw_vid.n_frames, enc_info.raw_vid.fps, show_str(blk_cnt),...
      toc(obj.file_start_time));

  end
  
  function open_analysis_output_files(obj, enc_info)
    
    if enc_info.do_inp_anls && isfield(enc_info.fdef, 'inp_mark')
      obj.inp_mark_blks = ...
        VidBlocksOut(enc_info.vid_blocker, enc_info.fdef.inp_mark, false, ...
        VidBlocker.BLK_STT_ZEXT);
      obj.inp_mark_blks.use_gpu = enc_info.proc_opts.use_gpu;
      obj.inp_mark_blks.use_single = enc_info.proc_opts.use_single;
    else
      obj.inp_mark_blks = [];
    end
    if enc_info.do_encode && isfield(enc_info.fdef, 'inp_vmark')
      obj.inp_vmark_blks = ...
        VidBlocksOut(enc_info.vid_blocker, enc_info.fdef.inp_vmark, false, ...
        VidBlocker.BLK_STT_ZEXT);
      obj.inp_vmark_blks.use_gpu = enc_info.proc_opts.use_gpu;
      obj.inp_vmark_blks.use_single = enc_info.proc_opts.use_single;
    else
      obj.inp_vmark_blks = [];
    end
    if enc_info.do_encode && isfield(enc_info.fdef, 'enc_mark')
      obj.enc_mark_blks = ...
        VidBlocksOut(enc_info.vid_blocker, enc_info.fdef.enc_mark, false, ...
        VidBlocker.BLK_STT_ZEXT);
      obj.enc_mark_blks.use_gpu = enc_info.proc_opts.use_gpu;
      obj.enc_mark_blks.use_single = enc_info.proc_opts.use_single;
    else
      obj.enc_mark_blks = [];
    end
    flds = {'inp_anls', 'inp_vanls', 'enc_anls';...
      'input analysis', 'input analysis by vector shifts', ...
        'encoder measurements analysis'; ...
      [], [], [];...
      'inp_sav', 'inp_vsav', 'enc_sav'; ...
      [], [], []};
    if enc_info.do_inp_anls
      bgn = 1;
    else
      bgn = 3;
    end
    if enc_info.do_encode
      last=3;
    else
      last=2;
    end
    for k=bgn:last
      if isfield(enc_info.fdef, flds{1,k})
        cso = CSVOutFile(enc_info.fdef.(flds{1,k}), ...
          VidBlocker.getVidInfoFields(), 'video info', 1);
        cso.writeRecord(enc_info.vid_blocker.getVidInfo());
        cso.setCSVOut(BlkMotion.csvProperties(), flds{2,k});
        flds{3,k} = cso;
      end
      if isfield(enc_info.fdef, flds{4,k})
        flds{5,k} = SAVOutFile(enc_info.fdef.(flds{4,k}),...
          enc_info.vid_blocker);
        
      end
    end
    obj.inp_anls = flds{3,1};
    obj.inp_vanls = flds{3,2};
    obj.enc_anls = flds{3,3};
    obj.inp_sav = flds{5,1};
    obj.inp_vsav = flds{5,2};
    obj.enc_sav = flds{5,3};
  end
  
  function cs_vid_io = run(obj, enc_info, decoder)
    % (Encoder/Decoder) This is the main "workhorse" function.
    % It does both CS encoding and decoding.  Specifically, it:
    % 1) reads in the video
    % 2) breaks the video into blocks
    % 3) for each block,
    %   --> takes compressed sensed measurements
    %   --> quantizes measurments
    %   --> arithmetic encodes measurments
    %   --> decodes symbols of arithmetic encoding
    %   --> reconstructs the signal using l1 minimization
    % 4) deblocks the video (puts blocks back together)
    %
    % Data on how each block was processed and the results of the processing
    % are stored in a list of CSVideoBlockProcessingData objects.
    %
    % Three files are written out each time this function is run.
    %
    % The output variable "csvideocodecinputoutputdata" in written out
    % in both binary format (.mat) and text format (.txt)
    % and the output video is written out in .yuv format.
    %
    % Note that there are three different files written out with the same
    % name but different extensions.
    %
    % The naming convention used for both writing out the output
    % variable and the output video is as follows:
    %
    % (test_video_index)vl
    % (number_of_frames_to_read_in)f
    % (entropy coding strategy)
    % (what to do with values outside quantizer range)
    % (100*vidmsrmnt_input_ratio)rt
    % (10*qntzr_ampl_stddev)std
    %
    % For example,
    %
    % 1v60fquantAndArEncodediscardLarge45rt30std10normQStep
    %
    % indicates that test video 1 was used (News), 60 frames were read
    % in, CS measurements were quantized and arithmetic encoded, CS
    % measurements outside range of quantizer were discared, the ratio
    % of CS measurements to input values was .45, 3 standard deviations
    % were preserved by the quantizer range and the quantization step
    % was 10 times the size of the normalized quantization step.
    %
    % [Input]
    %
    % obj - CS this codec object
    % fdef - A struct as produced by FilesDef.getFiles() defining I/O files.
    %
    % [Output]
    % cs_vid_io - a CSVideoCodecInputOutputData object which defines the input
    %             parameters and returns the output of the simulation.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    prefix = obj.proc_opts.prefix;
    
    blk_cnt = enc_info.vid_blocker.blk_cnt;
    
    base_len = 0;
    if enc_info.do_encode
      % Write preliminary information out
      base_elmnts = {enc_info.enc_opts, enc_info.raw_vid};
      for k=1:length(base_elmnts)
        base_len = base_len + base_elmnts{k}.codeLength(enc_info);
      end
    end
        
    n_blk_data_enc = 0; % no. of blocks encoded
    if ~all(blk_cnt)
      nxt_blk_indx = [];
    else
      nxt_blk_indx = [1,1,1];
    end
    
    pb_data = [];
    ttl_pxls = 0;

    if enc_info.proc_opts.report_frm
      frm_info = FrmReportInfo;
    else
      frm_info = [];
    end
    file_info = FileReportInfo;
    
    if obj.n_parblk > 0
      n_blks_data = obj.n_parblk;
    else
      n_blks_data = 1;
    end
      
    blks_data = struct(...
      'blk_time', cell(1,n_blks_data),...
      'enc_time', cell(1,n_blks_data),...
      'dec_time', cell(1,n_blks_data),...
      'ttl_time', cell(1,n_blks_data),...
      'last_in_frm', cell(1,n_blks_data),...
      'enc_info', cell(1,n_blks_data),...
      'xvec', cell(1,n_blks_data),...
      'vid_region', cell(1,n_blks_data),...
      'clr_blk', cell(1,n_blks_data),...
      'pre_diff_blk', cell(1,n_blks_data),...
      'tst_pre_diff_blk', cell(1,n_blks_data),...
      'enc_o', cell(1,n_blks_data),...
      'n_msrs', cell(1,n_blks_data),...
      'enc_cs_msrs', cell(1,n_blks_data),...
      'q_msr', cell(1,n_blks_data),...
      'len_msrs', cell(1,n_blks_data),...
      'len_enc', cell(1,n_blks_data),...
      'qmsr_snr', cell(1,n_blks_data),...
      'is_bgrnd', cell(1,n_blks_data),...
      'bgrnd_age', cell(1,n_blks_data),...
      'im_blk', cell(1,n_blks_data),...
      'imv_blk', cell(1,n_blks_data),...
      'em_blk', cell(1,n_blks_data)...
      );
    
    n_blks_decoded = 0;

    function [n_m, q_m, m_s] = get_n_msr_q_msr(iblk)
      prms = q_msrs.params(iblk);
      n_m = prms.n_clip + prms.n_no_clip;
      q_m = {q_msrs.getSingleBlk(iblk); };
      if nargout > 2
        m_s = { msrs(prms.clip_offset+prms.no_clip_offset+1:...
          prms.clip_offset+prms.no_clip_offset+prms.n_clip+prms.n_no_clip); };
      end
    end
    
    % Loop on all blocks and process them
    while ~isempty(nxt_blk_indx)
      blk_start_time = tic;
      
      cur_blk_indx = nxt_blk_indx;
      if obj.n_parfrm > 0
        [xvec, nxt_frm_indx, n_frmblk_read] = obj.raw_vid_in.getFrmBlks(...
          cur_blk_indx(3), cur_blk_indx(3) + obj.n_parfrm-1);
%         [nxt_frm_indx, xvec, n_frmblk_read] = ...
%           obj.getNextBlkFrms(cur_blk_indx(3), obj.n_parfrm, obj.raw_vid_in);
        if isempty(nxt_frm_indx)
          nxt_blk_indx = [];
          n_blks_data = n_frmblk_read*blk_cnt(1)*blk_cnt(2);
          blks_data = blks_data(1:n_blks_data);
        else
          nxt_blk_indx = [1,1,nxt_frm_indx];
        end
      else
        [xvec, nxt_blk_indx] = obj.raw_vid_in.getBlks(cur_blk_indx);
      end
      
      obj.raw_vid_in.discardFrmsBeforeBlk(nxt_blk_indx);

      [blks_data, xvec, enc_info, n_pxls] = ...
        obj.setBlocks(blks_data, xvec, enc_info, cur_blk_indx, decoder); 
      
      ttl_pxls = ttl_pxls + n_pxls;
      
      if ~isempty(frm_info)
        frm_info.update(n_pxls, blks_data);
      end
      file_info.update(n_pxls, blks_data);
        
      enc_len_computed = false;
      
      if enc_info.do_encode
        enc_start_time = tic;
      
        [msrs, q_msrs, sens_mtrx] = obj.encodeBlocks(blks_data, xvec);
         
        if ~obj.proc_opts.no_stats || ~isempty(obj.anls_opts) ||...
            (obj.proc_opts.report_blk && obj.proc_opts.report_qmsr_snr)
            
          if (obj.proc_opts.report_blk && obj.proc_opts.report_qmsr_snr)
            [n_b_msrs, b_q_msrs, b_msrs] = arrayfun(@get_n_msr_q_msr,1:n_blks_data);
              
            for iblk = 1:n_blks_data
              blks_data(iblk).n_msrs = n_b_msrs(iblk);
              blks_data(iblk).enc_info.q_msr = b_q_msrs{iblk};
              blks_data(iblk).enc_info.enc_cs_msrs = b_msrs{iblk};
              
              unq = enc_info.quantizer.unquantize(blks_data(iblk).enc_info.q_msr);
              blks_data(iblk).enc_info.vid_region = ...
                enc_info.vid_region.getSingleBlk(iblk);
              blks_data(iblk).sens_mtrx = ...
                obj.sns_mtrx_mgr.getBlockMtrx(blks_data(iblk).enc_info.blk_indx, ...
                blks_data(iblk).enc_info.vid_region);
              unq = blks_data(iblk).sens_mtrx.unsortNoClip(unq);
              orig_msrs = blks_data(iblk).enc_info.enc_cs_msrs;
              err_cs_msrmnts = unq - orig_msrs;
              q_snr = 20*log10((1e-3 + norm(orig_msrs))/...
                (1e-3 + norm(err_cs_msrmnts)));
              blks_data(iblk).qmsr_snr = sprintf(' Qmsr SNR=%4.1fdB', q_snr);
            end
          else
            [n_b_msrs, b_q_msrs] = arrayfun(@get_n_msr_q_msr,1:n_blks_data);
              
            for iblk = 1:n_blks_data
              blks_data(iblk).n_msrs = n_b_msrs(iblk);
              blks_data(iblk).enc_info.q_msr = b_q_msrs{iblk};
            end
          end

          if obj.proc_opts.report_qmsr_snr
            unq = enc_info.quantizer.unquantize(q_msrs);
            unq = sens_mtrx.unsortNoClip(unq);
            if ~isempty(frm_info)
              frm_info.updateMsrsErr(msrs, unq);
            end
            file_info.updateMsrsErr(msrs, unq);  
          end
        end

        if obj.proc_opts.check_coding || ~isempty(obj.enc_out)
          %            blks_data(1).len_msrs = codeLength(q_msrs{1}, enc_info);
          len_blk_msrs = codeLength(q_msrs, enc_info) / n_blks_data;
          for iblk=1:n_blks_data
            blks_data(iblk).len_msrs = len_blk_msrs;
            blks_data(iblk).len_enc = blks_data(iblk).len_msrs;
          end
          blks_data(1).len_enc = blks_data(1).len_msrs + base_len + ...
            codeLength(enc_info.vid_region, enc_info);
          enc_len_computed = true;
        end
        
        enc_time = toc(enc_start_time) / n_blks_data;
        
        if obj.proc_opts.check_coding || ~isempty(obj.enc_out)
          enc_elmnts = [ base_elmnts(:); {enc_info.vid_region; q_msrs(:)} ];
          base_elmnts = {};
          base_len = 0;
          
          [out_len, enc_info] = CSVidCodec.encOutputElmnts(enc_elmnts(:),...
            enc_info, obj.enc_out);
          
          if ~isempty(frm_info)
            frm_info.updateBytes(out_len, length(msrs));
          end
          file_info.updateBytes(out_len, length(msrs));
          
          if obj.proc_opts.check_coding && ~isempty(decoder)
            dec_src = obj.dec_inp;
          else
            dec_src = enc_elmnts;
          end
        end
      else
        enc_time = 0;
      end
      
      if ~enc_len_computed
        for iblk = 1:n_blks_data
          blks_data(iblk).len_msrs = 0;
          blks_data(iblk).len_enc = 0;
        end
      end
      
      if ~isempty(obj.anls_opts)
        for iblk = 1:length(blks_data)
          enc_start_time = tic;
          
          enc_o = obj.analyzeBlock(blks_data(iblk));
          
          blks_data(iblk).enc_time = ...
            blks_data(iblk).enc_time + toc(enc_start_time);
        end
        
        for iblk = 1:length(blks_data)
          
          blkd = blks_data(iblk);
          blk_info = blkd.enc_info;
          
          % Save background results
          if isfield(enc_o, 'is_bgrnd')
            blkd.is_bgrnd = enc_o.is_bgrnd;
            blkd.bgrnd_age = enc_o.bgrnd_age;
          end
          
          % Save m motion
          obj.save_motion(blk_info, enc_o.m_motion, ...
            obj.enc_anls, obj.enc_sav);
          
          if ~isempty(enc_o.m_motion) 
            if ~isempty(obj.enc_anls)
              enc_o.m_motion.setBlkInfo(...
                blk_info.vid_blocker.getBlkInfo(blk_info.blk_indx))
              obj.enc_anls.writeRecord(enc_o.m_motion.getCSVRecord());
            end
            
            if isfield(blk_info.fdef, 'enc_mark') 
              obj.enc_mark_blks = blk_info.vid_region.putIntoBlkArray(...
                enc_o.em_blk, obj.enc_mark_blks);
              err_msg = ...
                obj.enc_mark_blks.writeReadyFrames();
              if ischar(err_msg);
                error('failed writing enc_mark frames: %s', err_msg);
              end
            end
          end
          
          % Save x motion
          obj.save_motion(blk_info, enc_o.x_motion, ...
            obj.inp_anls, obj.inp_sav);
          
          if isfield(blk_info.fdef, 'inp_mark') && ~isempty(enc_o.x_motion)
            obj.inp_mark_blks = blk_info.vid_region.putIntoBlkArray(...
              enc_o.im_blk, obj.inp_mark_blks);
            err_msg = ...
              obj.inp_mark_blks.writeReadyFrames();
            if ischar(err_msg);
              error('failed writing inp_mark frames: %s', err_msg);
            end
          end
          
          % save v_motion
          obj.save_motion(blk_info, enc_o.v_motion, ...
            obj.inp_vanls, obj.inp_vsav);
          
          if isfield(enc_info.fdef, 'inp_vmark') && ~isempty(enco.v_motion)
            obj.inp_vmark_blks = blk_info.vid_region.putIntoBlkArray(...
              enc_o.imv_blk, obj.inp_vmark_blks);
            err_msg = ...
              obj.inp_vmark_blks.writeReadyFrames();
            if ischar(err_msg);
              error('failed writing inp_vmark frames: %s', err_msg);
            end
          end
          
          blkd.enc_info = blk_info;
          blks_data(iblk) = blkd;
        end
      end
      
      if ~obj.proc_opts.no_stats
        for iblk = 1:length(blks_data)
          % Save block information
          n_blk_data_enc = n_blk_data_enc + 1;
          obj.blk_data_list{n_blk_data_enc} =...
            CSVideoBlockProcessingData(blks_data(iblk));
        end
      end
      
      blk_time = toc(blk_start_time) / n_blks_data;
      
      if ~isempty(decoder)
        for iblk = 1:n_blks_data
          blks_data(iblk).blk_time = blk_time;
          blks_data(iblk).enc_time = enc_time;
        end
        
        pb_data = [ pb_data(:); blks_data(:)];
        
        if obj.proc_opts.check_coding==2 && ~isempty(decoder)
          tests = struct('enc_elmnts', {enc_elmnts}, 'indx', 1);
        else
          tests = [];
        end
          
        n_dec = 0;
        for iblk = 1:length(blks_data)
          % Decoding
          [rdata, tests] = obj.decodeBlock(decoder, dec_src, tests);
          
          [ndata, ttl_time] = obj.process_rdata(...
            rdata, pb_data, blk_time, enc_time, enc_info);
          
          n_dec = n_dec + ndata;
          
          if ~isempty(frm_info)
            frm_info.updatePsnr(rdata);
          end
          file_info.updatePsnr(rdata);
          
          if ~obj.proc_opts.no_stats
            for k= 1:ndata;
              obj.blk_data_list{n_blks_decoded+k}.seconds_to_process_block=...
                ttl_time(k);
            end
          end
          
        end
        
        n_blks_decoded = n_blks_decoded + n_dec;   
        pb_data = pb_data(n_dec+1:end);
        
      else
        if obj.proc_opts.report_blk
          for iblk = 1:n_blks_data
            blk_dur_str = sprintf('e=%6.3f(%6.3f)', blk_time, enc_time);
          
            obj.blk_report(blks_data(iblk), blk_dur_str, '');
          end
        end
      end
      
      if ~isempty(frm_info)
        frm_info.report(prefix);
      end
      
    end  % while on blocks
        
    if ~isempty(decoder)
      [rdata, msqr_err, psnr ]=decoder.finish();
      obj.process_rdata(rdata, pb_data, 0, 0, enc_info);
    end
    if ~isempty(frm_info)
      frm_info.report(prefix);
    end
    
    if ~obj.proc_opts.no_stats
      %calculate stats of the video blocks and store them in the
      %output
      cs_vid_io = CSVideoCodecInputOutputData(enc_info.enc_opts, enc_info.raw_vid);
      cs_vid_io.calculate(obj.blk_data_list(1:n_blk_data_enc));
      if ~isempty(decoder)
        cs_vid_io.msqr_err = msqr_err;
        cs_vid_io.psnr = psnr;
      end
      
      %save the output variable both in binary format and text format
      if isfield(enc_info.fdef, 'mat')
        save(enc_info.fdef.mat, 'cs_vid_io', '-mat');
      end
      if isfield(enc_info.fdef, 'txt')
        cs_vid_io.writeToTextFile(enc_info.fdef.txt);
      end
    else
      cs_vid_io = 'no_stats';
    end
    
    if ~isempty(decoder)  && isfield(enc_info.fdef, 'output')
      if isfield(enc_info.fdef, 'enc_vid')
        fprintf('%s DONE: %s ==>\n%s   %s ==>\n%s   %s\n', ...
          prefix, enc_info.fdef.input, prefix, ...
          enc_info.fdef.enc_vid, prefix, enc_info.fdef.output);
      else
        fprintf('%s DONE: %s ==>\n%s   %s\n', ...
          prefix, enc_info.fdef.input, prefix, enc_info.fdef.output);
      end
    elseif isfield(enc_info.fdef, 'enc_vid')
      fprintf('%s DONE: %s ==>\n%s   %s\n', ...
        prefix, enc_info.fdef.input, prefix, enc_info.fdef.enc_vid);
    else
      fprintf('%s DONE: %s \n', ...
        prefix, enc_info.fdef.input);
    end
    file_info.report(prefix);
    
  end
  
  function [blks, xvec, enc_info, n_pxls] = setBlocks(...
      obj, blks, xvec, enc_info, first_blk_indx, decoder)
    % setBlocks does some initial setting on a pixel blocks
    % Input:
    %   obj - this object
    %   blks - a struct array of n_blks blks_data structs to be updated
    %   xvec - vector of samples of the block
    %   enc_info - CSVidCodec state information
    %   first_blk_indx - index of first block
    %   decoder - used only as a flag
    % Output:
    %   blks - updated blks
    %   xvec - updated xvec (if blk_pre_diff)
    %   n_pxls - total raw pixels read
    
    n_blks = length(blks);
    
    pxlmx = enc_info.raw_vid.getPixelMax();
    blk_cnt = enc_info.vid_blocker.blk_cnt;
    
    if n_blks > 1
      blk_indx = obj.blk_z_indx(1:n_blks,:);
      blk_indx(:,3) = blk_indx(:,3) + first_blk_indx(3);
    else
      blk_indx = first_blk_indx;
    end
    
    n_pxls = 0;
    
    % Create vid_region
    zero_ext = struct('b',enc_info.enc_opts.zero_ext_b, ...
      'f', enc_info.enc_opts.zero_ext_f, 'w', enc_info.enc_opts.wrap_ext);
    enc_info.vid_region = VidRegion(blk_indx, enc_info.vid_blocker, zero_ext);

    % Set blocks.
    for iblk = 1:n_blks
      blk_info = enc_info;
      blk_info.blk_indx = blk_indx(iblk,:);
      blks(iblk).last_in_frm = ...
        all(blk_indx(iblk,1:2) == enc_info.vid_blocker.blk_cnt(1:2));
      
      cur_sptl_indx = blk_indx(iblk,1) + (blk_indx(iblk,2)-1) * blk_cnt(1);
      n_pxl_in_rgn = obj.n_raw_pxls(cur_sptl_indx);
      n_pxls = n_pxls + n_pxl_in_rgn;
      blk_info.n_pxl_in_rgn = n_pxl_in_rgn;
        
      blks(iblk).qmsr_snr = '';
      blks(iblk).len_msrs = 0;
      blks(iblk).len_enc = 0;
      blks(iblk).enc_info = blk_info;
    end

    clr_blk_needed = ...
      enc_info.do_inp_anls || ...
      any(enc_info.enc_opts.blk_pre_diff) ||...
      (isfield(enc_info.fdef, 'inp_copy') && isempty(blkd.clr_blk)) ||...
      (enc_info.do_encode && isfield(enc_info.fdef, 'inp_vmark')) ||...
      (enc_info.do_encode && isfield(enc_info.fdef, 'enc_mark'));
    
    xvec_needed = clr_blk_needed ||...
      ~isempty(enc_info.anls_opts) && ...
      isfield(enc_info.fdef,'inp_vmark') ||...
      isfield(enc_info.fdef,'inp_vanls') || isfield(enc_info.fdef,'inp_vsav');
    
    if xvec_needed
      if n_blks > 0
        for iblk = 1:n_blks
          blks(iblk).xvec = xvec(obj.blk_bgn(iblk):obj.blk_end(iblk));
        end
      else
        blks.xvec = xvec;
      end
      
      if clr_blk_needed
        clr_blks = enc_info.vid_region.pixelize(xvec, VidBlocker.BLK_STT_ZEXT);
        for iblk = 1:n_blks
          blks(iblk).clr_blk = clr_blks(:,iblk);
        end
        
        if any(enc_info.enc_opts.blk_pre_diff)
          pre_diff_blks = enc_info.vid_region.do_multiDiffExtnd(clr_blks,...
            enc_info.enc_opts.blk_pre_diff);
          obj.enc_pre_diff_blks = enc_info.vid_region.putIntoBlkArray(...
              pre_diff_blks, obj.enc_pre_diff_blks);
          
          for iblk = 1:n_blks
            blks(iblk).pre_diff_blk = pre_diff_blks(:,iblk);
          end
          
          if enc_info.tst_pre_diff
            tst_pre_diff_blk_s = enc_info.vid_region.do_multiDiffUnExtnd(...
              pre_diff_blks, enc_info.enc_opts.blk_pre_diff);
            obj.tst_pre_diff_blks = ...
                enc_info.vid_region.putIntoBlkArray(...
                tst_pre_diff_blk_s, obj.tst_pre_diff_blks);
              
            for iblk = 1:n_blks
              blks(iblk).tst_pre_diff_blk = tst_pre_diff_blk_s(:,iblk);
            end

            err_msg = ...
              obj.tst_pre_diff_blks.writeReadyFrames();
            if ischar(err_msg);
              error('failed writing tst_pre_diff frames: %s',...
                err_msg);
            end

          else
            for iblk = 1:n_blks
              blks(iblk).tst_pre_diff_blk = [];
            end
          end
  
          [nfrm, pre_diff_vid] = ...
            obj.enc_pre_diff_blks.writeReadyFrames();
          if ischar(nfrm);
            error('failed writing enc_pre_diff frames: %s', nfrm);
          end
          if nfrm && ~isempty(decoder)
            decoder.addRefPreDiff(pre_diff_vid);
          end
                    
        else   % blk_pre_diff not needed
          for iblk = 1:n_blks
            blks(iblk).pre_diff_blk = [];
            blks(iblk).tst_pre_diff_blk = [];
          end
        end
        
        if ~isempty(obj.inp_copy)
          for iblk = 1:n_blks
            obj.inp_copy.insertBlk(blks(iblk).clr_blk, ...
              blks(iblk).enc_info.blk_indx);
          end
          obj.inp_copy.writeReadyFrames();
        end
        
        if enc_info.do_inp_anls && isfield(enc_info.fdef, 'inp_mark')
          for iblk = 1:n_blks
            blks(iblk).im_blk = mark_blk_boundaries(blks(iblk).clr_blk,...
              enc_info.vid_blocker.ovrlp, enc_info.enc_opts.conv_rng, pxlmx);
          end
        else
          for iblk = 1:n_blks
            blks(iblk).im_blk = [];
          end
        end
        
        if enc_info.do_encode && isfield(enc_info.fdef, 'inp_vmark')
          for iblk = 1:n_blks
            blks(iblk).imv_blk = mark_blk_boundaries(blks(iblk).clr_blk,...
              enc_info.vid_blocker.ovrlp, enc_info.enc_opts.conv_rng, pxlmx);
          end
        else
          for iblk = 1:n_blks
            blks(iblk).imv_blk = [];
          end
        end
        
        if enc_info.do_encode && isfield(enc_info.fdef, 'enc_mark')
          for iblk = 1:n_blks
            enc_info.em_blk = mark_blk_boundaries(blks(iblk).clr_blk,...
              enc_info.vid_blocker.ovrlp, enc_info.enc_opts.conv_rng, pxlmx);
          end
        else
          for iblk = 1:n_blks
            blks(iblk).em_blk = [];
          end
        end
      else    % clr_blk not needed
        for iblk = 1:n_blks
          blks(iblk).clr_blk = [];
          blks(iblk).pre_diff_blk = [];
          blks(iblk).tst_pre_diff_blk = [];
          blks(iblk).im_blk = [];
          blks(iblk).imv_blk = [];
          blks(iblk).em_blk = [];
        end
      end
    else
      for iblk = 1:n_blks
        blks(iblk).xvec = [];
        blks(iblk).clr_blk = [];
        blks(iblk).pre_diff_blk = [];
        blks(iblk).tst_pre_diff_blk = [];
        blks(iblk).im_blk = [];
        blks(iblk).imv_blk = [];
        blks(iblk).em_blk = [];
      end
    end
      
    
    if any(enc_info.enc_opts.blk_pre_diff)
      b_xvec = cell(n_blks,1);
      for iblk = 1:n_blks
        b_xvec{iblk} = blks(iblk).pre_diff_blk;
      end
      xvec = vertcat(b_xvec);
    end
  end
  
  function [msrs, q_msrs, sens_mtrx] = encodeBlocks(obj, blks_data, xvec)
    
    if ~blks_data(1).enc_info.do_encode
      msrs = []; q_msrs = [];
      return
    end
    
    bidx = blks_data(1).enc_info.blk_indx;
    if obj.n_parfrm > 0
      sens_mtrx = obj.sns_mtrx_mgr.getFrmsMtrx(bidx(3),...
        blks_data(end).enc_info.blk_indx(3));
      
      msrs = sens_mtrx.multVec(xvec);
      nnclp = sens_mtrx.nNoClip();
      frm_msrs = sens_mtrx.sortNoClip(msrs);
      n_t_blks = blks_data(end).enc_info.blk_indx(3) - bidx(3)+ 1;
      q_msrs = blks_data(1).enc_info.quantizer.quantizeFrms(...
        frm_msrs(1:nnclp), frm_msrs(nnclp+1:end), n_t_blks);
      
    else
      sens_mtrx = obj.sns_mtrx_mgr.getBlockMtrx(blks_data.enc_info.blk_indx, ...
        blks_data.enc_info.vid_region);
      
      msrs = sens_mtrx.multVec(xvec);
      nnclp = sens_mtrx.nNoClip();
      blk_msrs = sens_mtrx.sortNoClip(msrs);
      q_msrs = blks_data.enc_info.quantizer.quantizeBlk(...
        blk_msrs(1:nnclp), blk_msrs(nnclp+1:end), blks_data.enc_info.blk_indx);
    end
    
%     % Get blocks measurements and quantize
%     for iblk=1:numel(blks_data)
%       msrs{iblk} = frm_msrs(msrs_bgn(iblk):msrs_end(iblk));
%       
%       enc_info = blks_data(iblk).enc_info;
%       intvl = QuantMeasurements.calcMsrsNoise(...
%         blks_data(iblk).enc_info.vid_region.n_pxls);
% %       blk_indx = enc_info.blk_indx;
%       
%       n_no_clip = mtrcs{iblk}.nNoClip();
%       q_msrs{iblk} = ...
%         blks_data(iblk).enc_info.quantizer.quantize(...
%         mtrcs{iblk}.sortNoClip(msrs{iblk}), intvl,...
%         struct('n_no_clip', n_no_clip,'mean',[], 'stdv',[]));
%       q_msrs{iblk}.arth_cdng = ...
%         (obj.enc_opts.lossless_coder == enc_info.enc_opts.LLC_AC);
%     end
%     msrs_ref = msrs;
%     q_msrs_ref = q_msrs;
    


%     [msrs_ref, q_msrs_ref] = arrayfun(@msrsQuant, blks_data, mtrcs,...
%       (1:numel(blks_data)));
%     
%     if ~iequal(msrs_ref, msrs)
%       error('msrs not equal');
%     end
%     if ~isEqual(q_msrs_ref, q_msrs)
%       error('q_msrs not equal');
%     end
%     if ~isEqual(msrs_ref, msrs)
%       error('measurments are different');
%     end
%     if ~isEqual(q_msrs_ref, q_msrs)
%       error('measurments are different');
%     end
    
%     function [msrs, qmsr] = msrsQuant(bdata, mtrx,iblk)
%       mtrx = mtrx{1};
%       msrs = frm_msrs(msrs_bgn(iblk):msrs_end(iblk));
%       indx = bdata.enc_info.blk_indx;
%       intvl = obj.q_intvl_blk(indx(1),indx(2));
%       n_no_clip = mtrx.nNoClip();
%       qmsr = ...
%         bdata.enc_info.quantizer.quantize(...
%         mtrx.sortNoClip(msrs), intvl,...
%         struct('n_no_clip', n_no_clip,'mean',[], 'stdv',[]));
%       qmsr.arth_cdng = ...
%         (obj.enc_opts.lossless_coder == enc_info.enc_opts.LLC_AC);
%       msrs = {msrs};
%       qmsr = {qmsr};
%       
%     end
  end
  
  function enc_out = analyzeBlock(obj, blkd)
    % Analyze a block
    
    enc_info = blkd.enc_info;
    blk_indx = enc_info.blk_indx;

    % Create vid_region
    zero_ext = struct('b',enc_info.enc_opts.zero_ext_b, ...
      'f', enc_info.enc_opts.zero_ext_f, 'w', enc_info.enc_opts.wrap_ext);
    vid_region = VidRegion(blk_indx, enc_info.vid_blocker, zero_ext);

    enc_out = struct();
    prefix = [obj.proc_opts.prefix show_str(enc_info.blk_indx)];
    sens_mtrx = [];
    
    % Compute background
    if obj.enc_opts.random.rpt_temporal  && enc_info.anls_opts.chk_bgrnd.mx_avg
      if isempty(sens_mtrx)
        sens_mtrx = obj.sns_mtrx_mgr.getBlockMtrx(blk_indx, vid_region);
      end
      
      [enc_o.is_bgrnd, enc_o.bgrnd_age] = ...
        obj.chk_bgrnd{blk_indx(1),blk_indx(2)}.checkBlk(...
            blk_indx(3), sens_mtrx, enc_info.enc_cs_msrs);
    end
    
    
    % Compute next frame cross correlations of input
    if enc_info.do_inp_anls
      [nrmxcor, enc_out.x_motion] = vid_region.nextFrmXCor(...
        blkd.clr_blk, VidBlocker.BLK_STT_ZEXT, enc_info.anls_opts);
      motion_found = enc_out.x_motion.motionFound()>0;
      if motion_found
        report_cor(nrmxcor, enc_out.x_motion, 'nrmd Xcor');
      end
      
      if ~isempty(blkd.im_blk)
        clr = [0, enc_info.raw_vid.getPixelMax()];
        position = [0.5, 0.5];
        enc_out.im_blk = vid_region.drawMotionMarker(...
          enc_info.im_blk, vid_region.BLK_STT_ZEXT, position, clr, ...
          enc_out.x_motion);
      end
    else
      enc_out.x_motion = [];
      motion_found = false;
    end

    if ~enc_info.do_encode
      enc_out.v_motion = [];
      enc_out.m_motion = [];
      return
    end
                
    % Compute cross correlations of (unquantized) measurements
    if ~isempty(enc_info.anls_opts) && ...
        (isfield(enc_info.fdef,'inp_mark') || isfield(enc_info.fdef,'inp_vmark') ||...
        isfield(enc_info.fdef,'inp_anls') || isfield(enc_info.fdef,'inp_sav') || ...
        isfield(enc_info.fdef,'inp_vanls') || isfield(enc_info.fdef,'inp_vsav'))
      
      if isfield(enc_info.fdef,'inp_vmark') ||...
          isfield(enc_info.fdef,'inp_vanls') || isfield(enc_info.fdef,'inp_vsav')
        % Computing v_motion
        if isempty(sens_mtrx)
          sens_mtrx = obj.sns_mtrx_mgr.getBlockMtrx(blk_indx, vid_region);
        end

        [nrmmcor, enc_out.m_motion, n_sum, nrmvcor, enc_out.v_motion] =...
          next_msrs_xcor(enc_out.enc_cs_msrmnts, sens_mtrx, ...
          vid_region, enc_info.anls_opts, blkd.xvec);
        
        if ~isempty(enc_info.imv_blk)
          clr = [0, enc_info.raw_vid.getPixelMax()];
          position = [0.5, 0.5];
          enc_out.imv_blk = vid_region.drawMotionMarker(...
            enc_info.imv_blk, vid_region.BLK_STT_ZEXT, position, clr, ...
            enc_out.v_motion);
        end
      else
        % Not computing v_motion
        [nrmmcor, enc_out.m_motion, n_sum] =...
          next_msrs_xcor(enc_info.enc_cs_msrmnts, blkd.sens_mtrx, ...
          vid_region, enc_info.anls_opts);
       
        enc_out.v_motion = [];
      end
      if ~isempty(nrmmcor)
        %              fprintf('%s number of terms: %d %d %d\n', prefix,...
        %                min(n_sum), mean(n_sum), max(n_sum));
        m_motion_found = enc_out.m_motion.motionFound()>0;
        if m_motion_found % && enc_info.anls_opts.ignore_edge
          % ignore motion in edge blocks
          blk_cnt = enc_info.vid_blocker.blk_cnt;
          m_motion_found = false;
          for k=1:size(vid_region.blk_indx,1)
            bi = vid_region.blk_indx(k,1:2);
            if all((bi>1) & (bi<blk_cnt(1:2)))
              m_motion_found = true;
              break;
            end
          end
        end
        if motion_found || m_motion_found
          if enc_info.anls_opts.chk_ofsts
            report_cor(nrmvcor, enc_out.v_motion, 'nrmd Vcor');
          end
          report_cor(nrmmcor, enc_out.m_motion, 'nrmd Mcor',n_sum);
        end
        
        if ~isempty(enc_info.em_blk)
          clr = [0, enc_info.raw_vid.getPixelMax()];
          position = [0.5, 0.5];
          enc_out.em_blk = vid_region.drawMotionMarker(...
            enc_info.em_blk, vid_region.BLK_STT_ZEXT, position, clr, ...
            enc_out.m_motion);
        end
      end
    else
      enc_out.m_motion = [];
      enc_out.v_motion = [];
    end
    
   function report_cor(xcor, info, name, n_sum)
      if nargin > 3
        n_mn = min(n_sum);
        n_mx = max(n_sum);
        if n_mn == n_mx
          sum_str = sprintf(' #terms=%d. ', n_mn);
        else
          sum_str = sprintf(' #terms=[%d %d %d]. ',...
            n_mn, mean(n_sum), n_mx);
        end
      else
        sum_str = '';
      end
      fprintf('%s %s %s%s\n', prefix, name, sum_str, info.report());
      
      % The following is test code which print detailed information
      % about the matches. To enables it remove the comment mark
      % below so that || 1 is on.
      if isempty(xcor)  % || 1
        for lvl=1:xcor.nStages()
          xc = xcor.getStageData(lvl);
          ofsts = xc.offsets;
          dstep = double(xc.step);
          fprintf('level %d step [%s]. zero match: %.4f best: %.4f at (%s)\n',...
            lvl, show_str(dstep), xc.mtch(1), xc.mx,...
            show_str(ofsts.numer(xc.mx_ind,:)));
          indcs = rat(ofsts);
          indcs1 = indcs(2:end,1:2);
          ind_bgn = min(indcs1);
          ind_end = max(indcs1);
          if all([0,0] <= ind_end) && all ([0,0] >= ind_bgn)
            ofsts_bgn = double(min(ofsts));
            indcs = indcs(:,1:2);
            mtc = xc.mtch;
          else
            indcs = indcs1;
            mtc = xc.mtch(2:end);
          end
          mtchs = -inf * ones(ind_end - ind_bgn + [2,2]);
          mtchs(2:end,1) = ofsts_bgn(1)+dstep(1)*((0:ind_end(1)-ind_bgn(1)))';
          mtchs(1,2:end) = ofsts_bgn(2)+dstep(2)*((0:ind_end(2)-ind_bgn(2)));
          ind_ofst = ind_bgn - [2,2];
          for j=1:size(indcs,1)
            indx = indcs(j,:) - ind_ofst;
            mtchs(indx(1),indx(2)) = mtc(j);
          end
          fprintf('%s\n', mtrx_to_str(mtchs));
        end
      end
    end
  end
  
end

methods(Static=true)
  
  function blk_report(blk_data, blk_dur, blk_psnr_str)
    if ~isempty(blk_data.n_msrs) && blk_data.n_msrs > 0
      b_msr_str = sprintf(' %.2f b/msr.', ...
        (8*blk_data.len_msrs)/double(blk_data.n_msrs));
    else
      b_msr_str = '';
    end
    fprintf('%s %s.%s Dur: %s sec.%s%s\n',...
      blk_data.enc_info.proc_opts.prefix, ...
      show_str(blk_data.enc_info.blk_indx), blk_data.qmsr_snr, ...
      blk_dur, b_msr_str, blk_psnr_str);
  end
  
%   function [nxt_frm_indx, xvec, n_frmblk_read] = ...
%       getNextBlkFrms(bgn_frm_indx, n_frms, raw_vid_in)
%     % getNextBlkFrms gets the blocks of several frames as a long vector
%     % Input:
%     %   cur_frm_indx - index of begining frame blocks to read.
%     %   n_frms - number of frame blocks to read.
%     %   raw_vid_in - VidBlocks object to read from
%     %   enc_info - CSVidCodec state information
%     % Output:
%     %   nxt_frm_indx - index of the next frame block to read, or empty if 
%     %     this frame of blocks was the last one.
%     %   xvec - vector of samples of all the blocks, concatenated.
%     %   clr_blk - If pixelization was necessary, a pixelization of xvec.
%     %             Othewise, it is empty
%     %   n_frmblk_read - no. of frme blocks actuall read
%     %   enc_info - updated CSVidCodec state information
%     %
%     
%     [xvec_ref, nxt_frm_indx_ref, n_frmblk_read_ref] = ...
%       raw_vid_in.getFrmBlks(bgn_frm_indx, bgn_frm_indx+n_frms-1);
% 
%     nxt_frm_indx = bgn_frm_indx;
%     frms = cell(1, n_frms);
%     n_frmblk_read = n_frms;
%     for k=1:n_frms
%       [frms{k}, nxt_frm_indx] = raw_vid_in.getBlks(nxt_frm_indx);
%       if isempty(nxt_frm_indx)
%         n_frmblk_read = k;
%         frms = frms(1:n_frmblk_read);
%         break;
%       end
%     end
%     xvec = vertcat(frms{:});
%         
%     if ~isequal(xvec_ref, xvec)
%       error('vxec not matching');
%     end
%     if ~isequal(nxt_frm_indx_ref, nxt_frm_indx)
%       error('nxt_frm_indx not matching');
%     end
%     if ~isequal(n_frmblk_read_ref, n_frmblk_read)
%       error('n_frmblk_read not matching');
%     end
%   end
   
  function [ndata, ttl_time] = process_rdata(rdata, pb_data, ...
      blk_time, enc_time, enc_info)
    ndata = length(rdata);
    ttl_time = zeros(ndata,1);
    
    for idx=1:ndata;
      dd = rdata{idx};
      dec_time = dd.read_dur + dd.proc_dur + dd.write_dur;
      ttl_time(idx) = pb_data(idx).blk_time + dec_time;
      
      if enc_info.proc_opts.report_blk
        blk_dur_str = sprintf('%7.3f e=%6.3f(%.3f) d=%7.3f(%7.3f)',...
          ttl_time, blk_time, enc_time, dec_time,  dd.proc_dur);
        
        if isfield(dd, 'blk_psnr')
          blk_psnr_str = sprintf(' PSNR=%.1f', dd.blk_psnr);
        else
          blk_psnr_str = '';
        end
        
        CSVidCodec.blk_report(pb_data(idx), blk_dur_str, blk_psnr_str);
      end
    end
  end
  
  function [len_enc, enc_info] = encOutputElmnts(enc_elmnts,...
      enc_info, enc_out)
    % Simulate encoding.
    % Input
    %   enc_elmnts - a cell array of the elemnts which need to be
    %       ded/decoded.
    %   enc_info - encoder info.  Must be present if enc_out is present
    %   enc_out - (optional).  If present and not empty the elements
    %       are encoded into this code destination.
    % Output
    %   len_enc - total length of elements (byte)
    %   enc_info - updated info
    
    if nargin < 3
      enc_out = [];
    end
    
    if isempty(enc_out)
      len_enc = 0;
      for k=1:length(enc_elmnts)
        len_enc = len_enc + enc_elmnts{k}.codeLength(enc_info);
      end
    else
      [len_enc, enc_info] = CodeElement.writeElements(...
        enc_info, enc_elmnts, enc_out);
      if ischar(len_enc)
        error('writing elements failed: %s', len_enc);
      end
    end
  end
  
  
  function save_motion(enc_info, motn, anls, sav)
    indx = enc_info.blk_indx;
    if ~isempty(motn) && (~isempty(anls) || ~isempty(sav))
      motn.setBlkInfo(enc_info.vid_blocker.getBlkInfo(indx));
      if ~isempty(anls)
        anls.writeRecord(motn.getCSVRecord());
      end
      if ~isempty(sav)
        sav.setBlkRecord(motn);
      end
    end
  end
    
  %**********************
  %  decodeBlock        *
  %**********************
  % Input
  %   decoder - Decoder object
  %   dec_inp - Can be either a CodeSource to read CodeElement objects
  %       from or a cell array of CodeElement objects.
  %   dec_info - info member of the decoder object
  %       decoded elements are compared to the encoded ones.
  %   tests - a struct specifying tests which need to be done. Can
  %       have any of the following fields:
  %         enc_info - if present correctness of decoding is checked
  %              (only if dec_inp is not a cell array)
  
  function [rdata, tests] = decodeBlock(decoder, dec_inp, tests)
    decoder.setReadStartTime();
    
    rdata = [];
    
    while true
      [dec_inp, code_elmnt, decoder.info, elmnt_len] = ...
        CSVidCodec.decInputElmnt(dec_inp, decoder.info);
      
      if isa(code_elmnt, 'CSVidFile') || isempty(code_elmnt)
        break;
      end
      
      if ~isempty(tests)
        if ~code_elmnt.isEqual(tests.enc_elmnts{tests.indx})
          error('Unequal decoding of %s',...
            class(tests.enc_elmnts{tests.indx}));
        end
        tests.indx = tests.indx + 1;
      end
      
      rdata = decoder.decodeItem(code_elmnt, elmnt_len);
      if ischar(rdata)
        error('%s', prefix, rdata);
      elseif ~isempty(rdata)
        break
      end
    end
  end
  
  % Simulate decoding.
  % Input
  %   dec_inp - Can be either a CodeSource to read CodeElement objects
  %       from or a cell array of CodeElement objects.
  %   dec_info - info member of the decoder object
  %       decoded elements are compared to the encoded ones.
  %   enc_info - (optional) relevant only if dec_inp is a CodeSource.
  %           Then, if present forces checking of correctness of CodeElements
  %           encoding.
  % Output
  %   dec_inp - updated dec_inp.  If a cell array the first element is
  %           removed
  %   code_elmnt - the decoded element
  %   dec_info - updated dec_info
  
  function [dec_inp, code_elmnt, dec_info, elmnt_len] = decInputElmnt(...
      dec_inp, dec_info)
    
    code_elmnt = [];
    elmnt_len = 0;
    if iscell(dec_inp)
      while ~isempty(dec_inp)
        if isa(dec_inp{1}, 'CodeElementTypeList')
          dec_info.type_list = dec_inp{1};
          dec_inp = dec_inp(2:end);
        else
          code_elmnt = dec_inp{1};
          dec_inp = dec_inp(2:end);
          elmnt_len = 0;
          return;
        end
      end
    else
      [code_elmnt, elmnt_len, dec_info] = ...
        CodeElement.readElement(dec_info, dec_inp);
      if ischar(code_elmnt)
        error(['Error in CodeElement:readElement(): ', code_elmnt]);
      elseif isscalar(code_elmnt) && code_elmnt == -1
        code_elmnt = [];
        return
      end
    end
  end
  
  function cs_vid_io = ...
      doSimulationCase(enc_opts, anls_opts, dec_opts, files_def,...
      proc_opts)
    
    % doSimulationCase() Runs one case of simulation on all files.
    %
    %
    % [Input]
    %
    % enc_opts - a CS_EncVidParams objects or something which can be used
    %            as an argument to construct such an object:
    %              A struct in which each field specify the property
    %                 to be changed.  An empty struct or empty array
    %                 may be used if no change to the defaults is
    %                 necessary.
    %              A JSON string specifying such a struct.  If a field
    %                 value is a string beginning with an ampersand (&), the field minus
    %                 the ampersand prefix is evaluated before assignment.
    %              A string containing an '<' followed by a file name.
    %                A JSON string is read from the file and converted
    %                to a struct as above.
    % anls_opts - If present and not empty speciifies options for
    %             measurements analysis. can be an object of type
    %             CS_AnlsParams or something which can be used as an
    %             argument to construct such an object:
    %              A struct in which each field specify the property to
    %              be
    %                 changed.  An empty struct or empty array may be
    %                 used if no change to the defaults is necessary.
    %              A JSON string specifying such a struct.  If a field
    %              value is
    %                 a string beginning with an ampersand (&), the
    %                 field minus the ampersand prefix is evaluated
    %                 before assignment.
    %              A string containing an '<' followed by a file name. A
    %              JSON
    %                string is read from the file and converted to a
    %                struct as above.
    % dec_opts -  if present and not empty perform decoding. dec_opts
    %            can be an object of type CS_DecParams or something which
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
    % files_def - A FilesDef object or a string which cotains a JSON...
    %             representation of a FilesDef object, or a name of
    %             a file containing a JSON object.  A file name is
    %             indicated by a '<' prefix.
    %             If files_def is a string it is interpreted as
    %                struct('name',files_def, 'output_id','*')
    % proc_opts (Optional) - a struct with any of the following
    %      fields which impact the processing mode (other fields are
    %      ignored):
    %         io_types - (optional) cell array of types to be processed.
    %         output_id - (optional) If present, the output directory is
    %                     set to output_id.  If output_id contains an
    %                     sterisk, the sterisk is replaced by a
    %                     date.
    %         case_id - format for call to enc_opts.idStr(). Default
    %                   ']'.
    %         prefix - (0ptional) and identifying prefix to add
    %                  before all printing. Default: '] '
    %         inp_anls - mode of input file analysis (default=1):
    %                        0 - never
    %                        1 - only if ~do_encode
    %                        2 - always
    %         par_files - If true process files in parallel.
    %                     Default: false.
    %         keep_sock - For the case that the input is a socket: If
    %                    defined and true, the socket is not closed upon
    %                    exit. Otherwise it is closed.
    %         use_gpu - If true, use GPU (default = false).
    %         use_single - If true, uss single precision (default = false)
    %         no_stats - If present and true, statistics are not computed,
    %         check_coding - relevant only when decoding is done. If 0
    %                        (defaut) code elements are lossless encoded for
    %                        computing length, but passed to the decoder as
    %                        is, so the decoder does not have to do the
    %                        lossless decoding. If 1, the losslessly
    %                        encoded code elemens are passed to the decoder
    %                        and losslessly decoded. 2 is the same as 1 and
    %                        in addition, the code elements decoded by the
    %                        decoder are compared to the original ones at
    %                        the encoder and an error is thrown if there is
    %                        a difference.
    %         prof_spec - profiling specification. allowed values are:
    %                       0 - no profiling
    %                       1 - profile only run() function.
    %                       2 - profile everything.
    %        report_qmsr_snr - If true report the SNR of measurements 
    %                          quantization error each block.
    %                          default: false
    %                     default: 0
    % [Output]
    % cs_vid_io - a CSVideoCodecInputOutputData object which defines the input
    %             parameters and returns the output of the simulation. If
    %             proc_opts.no_stat is true, the string "no_stats" is
    %             returned instead.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    case_start = tic;
    
    if nargin >= 5 && isfield(proc_opts, 'prof_spec');
      prof_ctrl(1, proc_opts.prof_spec);
    else
      prof_ctrl(1);
    end 
    
    def_proc_opts = struct(...
      'output_id', '*',...
      'case_id', '<Nn>',...
      'prefix', '] ',...
      'par_files', false,...
      'use_gpu', CompMode.defaultUseGpu(),...
      'use_single', false,...
      'no_stats', false,...
      'check_coding', 0,...
      'prof_spec', 0,...
      'report_qmsr_snr', false);
    
    if nargin >= 5
      flds = fieldnames(proc_opts);
      for k=1:length(flds)
        fld = flds{k};
        def_proc_opts.(fld) = proc_opts.(fld);
      end
    end
    proc_opts = def_proc_opts;

    if proc_opts.par_files && ~isempty(getCurrentTask())
      proc_opts.par_files = false;
    end
    
    if proc_opts.use_gpu && ~isfield(proc_opts, 'gpus')
      proc_opts.gpus = find_gpus();
      if isempty(proc_opts.gpus)
        proc_opts.use_gpu = false;
      end
    end
    if ~proc_opts.use_gpu && CompMode.defaultUseGpu()
      error('Specified no_gpu while MEX SW is compiled for GPU');
    end
    
    if ~isa(enc_opts, 'CS_EncVidParams')
      enc_opts = CS_EncVidParams(enc_opts);
    end
    prefix = enc_opts.idStr(proc_opts.prefix);
    proc_opts.prefix = prefix;
    if ~proc_opts.no_stats
      cs_vid_io = CSVideoCodecInputOutputData(enc_opts);
      cs_vid_io.clearResults();
    else
      cs_vid_io = 'no_stats';
    end
    
    if ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
      anls_opts = CS_AnlsParams(anls_opts);
    end
    
    if ~isempty(dec_opts) && ~isa(dec_opts, 'CS_DecParams')
      dec_opts = CS_DecParams(dec_opts);
    end
    
    proc_opts.case_id = enc_opts.idStr(proc_opts.case_id);
    
    % Initialize input files
    io_id = struct('dec', 0, 'case', proc_opts.case_id);
    
    if ischar(files_def) || isstruct(files_def)
      io_id.output = regexprep(proc_opts.output_id, '[*]',...
        datestr(now,'yyyymmdd_HHMM'));
      io_params = struct('identifier', io_id, 'do_enc', true);
      if isfield(proc_opts, 'io_types')
        io_params.fields = proc_opts.io_types;
      end
      files_def = FilesDef(files_def, io_params);
      fprintf('%s full_path_of_output=%s\n',prefix, ...
        files_def.getEncoderDir());
    else
      files_def.specifyDirs(io_id);
    end
    files_def.makeDirs();
    
    enc_dir = files_def.getEncoderDir();
    dec_dir = files_def.getDecoderDir();
    svn_ver = getSvnVersion();
    mex_clnp = initMexContext();  %#ok
    
    if proc_opts.use_gpu && ~isempty(proc_opts.gpus)
      init_gpu(proc_opts.gpus(1));
    end
    
    fprintf(['%s Starting simulation case: %s. SW version: %s\n'...
      '%s   output dir=%s\n'], ...
      prefix, proc_opts.case_id, svn_ver, prefix, enc_dir);
    
    write_str_to_file(enc_dir, 'version.txt', svn_ver);

    report_opts(enc_opts, 'Encoding options', 'enc_opts', {enc_dir});
    
    proc_opts_str = ...
      show_str(proc_opts, struct(), struct('prefix',prefix, 'struct_marked', true));
    fprintf('%s Processing options:\n%s\n', prefix, proc_opts_str);
    write_str_to_file(dec_dir, 'proc_opts.txt', proc_opts_str);
    mat2json(proc_opts, fullfile(dec_dir, 'proc_opts.json'));
    
    if ~isempty(anls_opts)
      report_opts(anls_opts, 'Analysis options', 'anls_opts', ...
        {enc_dir,dec_dir});
    end
    
    if ~isempty(dec_opts) || (~isempty(anls_opts) && (...
        files_def.isType('dec_anls') || files_def.isType('dec_sav')))
      report_opts(dec_opts, 'Decoding options', 'dec_opts', {dec_dir});
      
      switch proc_opts.check_coding
        case 0
          fprintf(...
            '%s Performing lossless encoding but no lossless decoding\n',...
            prefix);
        case 1
          fprintf(...
            '%s Performing lossless encoding and decoding\n',...
            prefix);
        case 2
          fprintf(...
            '%s Checking correctness of lossless encode/decode\n',...
            prefix);
      end
      
    end
    
    function report_opts(opts, name, fname, dirs)
      fprintf('%s %s:\n%s\n', prefix, name, opts.describeParams([prefix,'  '] ))
      opts_str = opts.describeParams();
      dirs = unique(dirs);
      for kk=1:length(dirs)
        dr = dirs{kk};
        write_str_to_file(dr, [fname, '.txt'], opts_str);
        
        opts.getJSON(fullfile(dr, [fname, '.json']));
      end
    end
    
    % Print file specificaiton info
%     dirs = unique({enc_dir, dec_dir});
%     for k=1:length(dirs)
%       J = mat2json(struct('files_specs',files_def.getAllFiles()),...
%         fullfile(dirs{k},'files_specs.json'));
%     end
    
    rnd_seed_step = 10000;
    if proc_opts.par_files && isempty(getCurrentTask()) && isempty(files_def.sock)
      pool = gcp('nocreate');
      if isempty(pool)
        pool_size = 0;
      else
        pool_size = pool.NumWorkers;
      end
    else
      pool_size = 0;
    end
    if pool_size
      fdef_list = files_def.getAllFiles();
      enc_opts_fdef = cell(size(fdef_list));
      proc_opts_fdef = cell(size(fdef_list));
      sprintf('%s%d]',prefix,k)
      cs_vid_fdef = cell(size(fdef_list));
      random = enc_opts.random;
      for k=1:length(fdef_list)
        enc_opts_fdef{k} = enc_opts;
        enc_opts_fdef{k}.setParams(struct('random',random));
        proc_opts_fdef{k} = proc_opts;
        proc_opts_fdef{k}.prefix = sprintf('%s%d]',prefix,k);
        proc_opts_fdef{k}.keep_sock = true;
        random.seed = random.seed+rnd_seed_step;
      end
      parfor k=1:length(fdef_list)
        codec = CSVidCodec( enc_opts_fdef{k}, anls_opts, dec_opts,...
          proc_opts_fdef{k});
        [enc_inf, dcdr] = codec.init(fdef_list(k));
        
        prof_ctrl(2, proc_opts_fdef{k}.prof_spec);
        
        cs_vid_fdef{k} = codec.run(enc_inf, dcdr);
        
        prof_ctrl(1, proc_opts_fdef{k}.prof_spec);
      end
      reset_fast_heap_mex();
      if ~proc_opts.no_stats
        for k=1:length(fdef_list)
          cs_vid_io.add(cs_vid_fdef{k});
        end
      end
    else
      indx = files_def.init_getFiles();
      k=1;
      while true
        [fdef, indx, base_name] = files_def.getFiles(indx);
        if isequal(fdef,[])
          break
        end
        
        if ~isempty(files_def.sock)
          [len_enc, ~] = CSVidFile(base_name).write(struct(), files_def.sock, true);
          if ischar(len_enc)
            error('Sending CSVidFile("%s") over socket failed: %s', ...
              base_name, len_enc);
          end
        end
        
        prc_opts = proc_opts;
        prc_opts.keep_sock = true;
        prc_opts.prefix = sprintf('%s%d]',prefix,k);
        codec = CSVidCodec( enc_opts, anls_opts, dec_opts,prc_opts);
        [enc_inf, dcdr] = codec.init(fdef);
        
        prof_ctrl(2, proc_opts.prof_spec);
        
        cs_vid_fdef = codec.run(enc_inf, dcdr);
        
        prof_ctrl(1, proc_opts.prof_spec);

        reset_fast_heap_mex();
        
        if ~isempty(files_def.sock)
          [len_enc, ] = CSVidFile('').write(struct(), files_def.sock, true);
          if ischar(len_enc)
            error('Sending CSVidFile("") over socket failed: %s', len_enc);
          end
        end
        
        k = k+1;
        if ~proc_opts.no_stats
          cs_vid_io.add(cs_vid_fdef);
        end
        
        % Changing seed for next file
        random = enc_opts.random;
        random.seed = random.seed+rnd_seed_step;
        enc_opts.setParams(struct('random', random));
        
      end
    end
    
    % Close output socket if necessary
    if ~isfield(proc_opts, 'keep_sock') || ~proc_opts.keep_sock
      CodeDest.deleteCodeDest(files_def.sock);
    end
    
    case_time = toc(case_start);
    fprintf('%s Case duration: %f\n', prefix, case_time);
    
  end % doSimulationCase()]
  
  function [simul_info]=doSimulation(enc_opts, anls_opts, dec_opts,...
      io_def, proc_opts)
    % runs encoding and possibly analysis and decoding on a set of
    % files in different encoding cases. The output is saved in .mat
    % file.
    %
    % Input:
    %   enc_opts - a specification of encoder options which override the
    %              defaults. It can have one of the following forms:
    %              A struct in which each field specify the property to be
    %                 changed.  An empty struct or empty array may be used if
    %                 no change to the defaults is necessary.
    %              A JSON string specifying such a struct.  If a field value is
    %                 a string beginning with an ampersand (&), the field minus
    %                 the ampersand prefix is evaluated before assignment.
    %              A string containing an '<' followed by a file name. A JSON
    %                string is read from the file and converted to a struct as
    %                above.
    %              Multiple cases may be defined in the method
    %              specified in ProcessingParams.getCases().
    %
    % anls_opts - If present and not empty speciifies options for
    %             measurements analysis. can be an object of type
    %             CS_AnlsParams or something which can be used as an
    %             argument to construct such an object:
    %              A struct in which each field specify the property to
    %              be
    %                 changed.  An empty struct or empty array may be
    %                 used if no change to the defaults is necessary.
    %              A JSON string specifying such a struct.  If a field
    %              value is
    %                 a string beginning with an ampersand (&), the
    %                 field minus the ampersand prefix is evaluated
    %                 before assignment.
    %              A string containing an '<' followed by a file name. A
    %              JSON
    %                string is read from the file and converted to a
    %                struct as above.
    % dec_opts - if empty, no decoding is done.  Otherwise, specifies decoder
    %            options which override the default. It can have one of
    %            the following forms:
    %              A struct in which each field specify the property to be
    %                 changed.  An empty struct or empty array may be used if
    %                 no change to the defaults is necessary.
    %              A JSON string specifying such a struct.  If a field value is
    %                 a string beginning with an ampersand (&), the field minus
    %                 the ampersand prefix is evaluated before assignment.
    %              A string containing an '<' followed by a file name. A JSON
    %                string is read from the file and converted to a struct as
    %                above.
    % io_def - A FilesDef object or a string which cotains a JSON...
    %             representation of a FilesDef object, or a name of
    %             a file containing a JSON object.  A file name is
    %             indicated by a '<' prefix.
    %             If files_def is a string it is interpreted as
    %                struct('name',files_def, 'output_id','*')
    % proc_opts (Optional) - a struct with any of the following
    %      fields which impact the processing mode (other fields are
    %      ignored):
    %         io_types - an optional cell array of the types to be
    %             processed (used as the optional argument fields in
    %             FilesDef constructor). This argument is ignored if
    %             io_def is a FilesDef object.
    %         output_id - a string suitable for specifying output_id.
    %             a '*' in the string is replaced by time and day.
    %         case_id - format for call to enc_opts.idStr(). Default
    %                   '<Nn>'.
    %         prefix - (0ptional) and identifying prefix to add
    %                  before all printing. Default '<Nn>] '
    %         inp_anls - mode of input file analysis (default=1):
    %                        0 - never
    %                        1 - only if ~do_encode
    %                        2 - always
    %         par_files - If true process files in parallel.
    %                     Default: false.
    %         par_cases - If non-zero, number of cases to process in
    %                     parallel.
    %         use_gpu - If true, use GPU (default = false).
    %         use_single - If true, uss single precision (default = false)
    %         no_stats - If present and true, statistics are not computed.
    %         check_coding - relevant only when decoding is done. If 0
    %                        (defaut) code elements are lossless encoded for
    %                        computing length, but passed to the decoder as
    %                        is, so the decoder does not have to do the
    %                        lossless decoding. If 1, the losslessly
    %                        encoded code elemens are passed to the decoder
    %                        and losslessly decoded. 2 is the same as 1 and
    %                        in addition, the code elements decoded by the
    %                        decoder are compared to the original ones at
    %                        the encoder and an error is thrown if there is
    %                        a difference.
    %         prof_spec - profiling specification. allowed values are:
    %                       0 - no profiling
    %                       1 - profile only run() function.
    %                       2 - profile everything.
    %                     default: 0
    %        report_qmsr_snr - If true report the SNR of measurements 
    %                          quantization error each block.
    %                          default: false
    %
    % [Output]
    %    simul_info - a struct containing the test conditions and
    %                 the results for each case.
    
    if nargin >= 5
      prof_ctrl(1, proc_opts);
    else
      prof_ctrl(1);
    end 
    
    def_proc_opts = struct(...
      'output_id', '*',...
      'case_id', '<Nn>',...
      'prefix', '<Nn>] ',...
      'par_files', false,...
      'par_cases', 0,...
      'use_gpu', CompMode.defaultUseGpu(),...
      'use_single', false,...
      'no_stats', false,...
      'check_coding', 0,...
      'prof_spec', 0,...
      'report_qmsr_snr', false);
    
    if nargin >= 5
      flds = fieldnames(proc_opts);
      for k=1:length(flds)
        fld = flds{k};
        def_proc_opts.(fld) = proc_opts.(fld);
      end
    end
    proc_opts = def_proc_opts;

    proc_opts.output_id = regexprep(proc_opts.output_id, '*',...
      datestr(now,'yyyymmdd_HHMM'));
        
    %place all simulation output in uniquely identifable directory
    % Initialize input files
    io_id = struct('output', proc_opts.output_id, 'dec', 0);
    if isa(io_def,'FilesDef')
      files_def = io_def;
      files_def.specifyDirs(io_id);
    else
      io_params = struct('identifier', io_id, 'do_enc', true);
      if isfield(proc_opts,'io_types')
        io_params.fields = proc_opts.io_types;
      end
      files_def = FilesDef(io_def, io_params);
    end
    files_def.makeOutputDir();
    
    prefix = regexprep(proc_opts.prefix, '[<].*[>]','');
    fprintf('%s CSVidCodec.doSimulation() SVN version: %s\n%s  Output_dir=%s\n',...
      prefix, getSvnVersion(), prefix, files_def.outputDir());
    
    % Initialize options
    if ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
      anls_opts = CS_AnlsParams(anls_opts);
    end
    if ~isempty(dec_opts) && ~isa(dec_opts, 'CS_DecParams')
      dec_opts = CS_DecParams(dec_opts);
    end
    
    simul_info = struct('files_def', files_def, 'anls_opts', anls_opts,...
      'dec_opts', dec_opts, 'n_done', 0, 'proc_opts', proc_opts);
    simul_info.case_list = ProcessingParams.getCases(enc_opts);
    simul_info.case_list = ProcessingParams.setCaseNo(simul_info.case_list);
    simul_info.sml_rslt = cell(size(simul_info.case_list));
    
    if simul_info.proc_opts.par_cases
      simul_info. proc_opts.par_files = false;
    end
    
    % Do the processing
    simul_info = CSVidCodec.continueSimulation(simul_info);
    
  end  % doSimulation
  
  % Continue a simulation started by doSimulation
  %   Input
  %     sml - either a struct read from .mat file or
  %           the file name
  function simul_info = continueSimulation(simul_info)
    if ischar(simul_info)
      simul_info = load(simul_info);
    end
    
    %start the stop watch
    simulation_start=tic;
    start_date = datestr(now,'yyyymmdd_HHMM');
    
    prof_ctrl(1, simul_info.proc_opts.prof_spec);

    n_cases = numel(simul_info.case_list);
    anls_opt_s = simul_info.anls_opts;
    dec_opt_s = simul_info.dec_opts;
    files_def = simul_info.files_def;
    proc_opt_s = simul_info.proc_opts;
    
    if proc_opt_s.use_gpu
      proc_opt_s.gpus = find_gpus();
      if isempty(proc_opt_s.gpus)
        proc_opt_s.use_gpu = false;
      end
    end
    if ~proc_opt_s.use_gpu && CompMode.defaultUseGpu()
      error('Specified no_gpu while MEX SW is compiled for GPU');
    end
    
    if proc_opt_s.par_cases && isempty(files_def.sock)
      pool = gcp('nocreate');
      if isempty(pool)
        pool_size = 0;
      else
        pool_size = pool.NumWorkers;
      end
    else
      pool_size = 0;
    end
      
    if pool_size
      case_proc_opts = cell(1,n_cases);
      for j=1:n_cases
        case_proc_opts{j} = proc_opt_s;
      end
      for j_case = 1:proc_opt_s.par_cases:n_cases
        end_case = min(j_case+proc_opt_s.par_cases-1, n_cases);
        n_par_cases = end_case- j_case+1;
        case_list = simul_info.case_list(j_case:end_case);
        sml_rslt = simul_info.sml_rslt(j_case:end_case);
        had_error = false(n_par_cases,1);
        parfor i_case =1:n_par_cases
          try
            if isempty(sml_rslt{i_case})
              sml_rslt{i_case} =CSVidCodec.doSimulationCase(...
                case_list{i_case}, anls_opt_s, dec_opt_s,...
                files_def, case_proc_opts{i_case});
            end
          catch exc
            had_error(i_case) = true;
            fprintf('**** case %d: Error: %s\n    %s\n', ...
              j_case+i_case-1, exc.identifier, exc.message);
            dbstack;
            
          end
        end
        if any(had_error)
          error('Error in parallel processing');
        end
        simul_info.sml_rslt(j_case:end_case) = sml_rslt;
        
        %save the result of the simulation so far
        save([simul_info.files_def.outputDir() 'simul_info.mat'], '-struct',...
          'simul_info', '-mat');
      end
    else
      prc_opts = proc_opt_s;
      prc_opts.keep_sock = true;
      for i_case = 1:n_cases
        enc_opt_s = CS_EncVidParams(simul_info.case_list{i_case});
        
        if isempty(simul_info.sml_rslt{i_case})
          if ~isempty(files_def.sock)
            case_id = enc_opt_s.idStr(prc_opts.case_id);
            files_def.specifyDirs(struct('case', case_id));
            case_dir = files_def.caseDir();
            
            [len_enc, ~] = ...
              CSVidCase(case_dir).write(struct, files_def.sock, true);
            if ischar(len_enc)
              error('Sending CSVidCase("%s") over socket failed: %s', ...
                files_def.caseDir(), len_enc);
            end
          end
            
          simul_info.sml_rslt{i_case} = CSVidCodec.doSimulationCase(...
            enc_opt_s, anls_opt_s, dec_opt_s, files_def, prc_opts);
          
          if ~isempty(files_def.sock)
            [len_enc, ~] = ...
              CSVidCase('').write(struct(), files_def.sock, true);
            if ischar(len_enc)
              error('Sending CSVidCase("%") over socket failed: %s', len_enc);
            end
          end
          
          %save the result of the simulation si far
          save([simul_info.files_def.outputDir() 'simul_info.mat'], ...
            '-struct', 'simul_info', '-mat');
        end
      end
    end
    
    %stop the stop watch
    simul_sec = toc(simulation_start);
    
    % Close output socket if necessary
    if ~isfield(proc_opt_s, 'keep_sock') || ~proc_opt_s.keep_sock
      CodeDest.deleteCodeDest(files_def.sock);
    end
    
    fprintf('\n   done. Start: %s. End: %s. %d sec.\n', start_date,...
      datestr(now,'yyyymmdd_HHMM'), simul_sec);
  end
  
end %methods (Static=true)
end
