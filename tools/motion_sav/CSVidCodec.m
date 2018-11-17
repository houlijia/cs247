classdef CSVidCodec
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
                
        %decide whether or not to do profiling
        profiling_setting = struct('on', {'','profile on'},...
            'viewer', {'','profile viewer'});
        
        % Can be 1 or 2 for profiling off or on, respectively
        do_profiling=1;
        
        % Check encoding/decoding.  Can get three values:
        %   0 - encoding is done only to compute length. Decoding is not
        %       done at all.  Instead, CodeElement objects are passed
        %       directly to the decoder.
        %   1 - Both encoding and decoding are done.
        %   2 - Both encoding and decoding are done and the decoded objects
        %       are compared to the original objects
        check_coding = 1;
        
        % If true report the SNR of measurements quantization error each
        % block
        report_qmsr_snr = false;
        
        % Control using parallel processing on blocks. If 0 no parallel
        % processing is done.  Otherwise this the maximal number of
        % blocks done in parallel is the workers pool size times this value
         parallel_blocks = 4;
%          parallel_blocks = 0;
        
%         report_msrmnt_xcor = false;
        report_msrmnt_xcor = true;

    end
    
   methods(Static=true)
       
       % (Encoder/Decoder) This is the main "workhorse" function.  It does both CS encoding and
       % decoding.  Specifically, it:
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
       % fdef - A struct as produced by FilesDef.getFiles() defining I/O files.
       % enc_opts - a CS_EncParams objects or something which can be used
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
       % prefix -   (optional) If prsent will preced each printed line
       % 
       % [Output]
       % cs_vid_io - a CSVideoCodecInputOutputData object which defines the input
       %             parameters and returns the output of the simulation. 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function cs_vid_io = run(fdef, enc_opts, anls_opts, dec_opts, prefix)
           
           pool = gcp('nocreate');
           if isempty(pool)
             n_parblk = 0;
           else
             n_parblk = CSVidCodec.parallel_blocks * pool.NumWorkers;
           end
           
           % Start the profile viewer
           eval(CSVidCodec.profiling_setting(...
               CSVidCodec.do_profiling).on);

           if ~isa(enc_opts, 'CS_EncParams')
               enc_opts = CS_EncParams(enc_opts);
           end

           if nargin < 5
               prefix = ']';
               if nargin < 4
                   dec_opts = [];
               end
           end

           if ~isempty(dec_opts) && ~(isfield(fdef, 'output') || ...
                   isfield(fdef, 'dec_mark') || isfield(fdef, 'dec_slct_mark') ||...
                   isfield(fdef, 'dec_anls') || isfield(fdef, 'dec_sav') || ...
                   isfield(fdef, 'dec_pre_diff') || ...
                   isfield(fdef, 'tst_pre_diff'))
               dec_opts = [];
           elseif isempty(dec_opts) && (~isempty(anls_opts) && (...
                   isfield(fdef,'dec_anls') || isfield(fdef,'dec_sav')))
               dec_opts = CS_DecParams();
               flds = fields(fdef);
               for k = 1:length(flds);
                   fld = flds{k};
                   if any(strcmp(fld, {'output', 'dec_mark', 'dec_slct_mark',...
                           'dec_anls', 'dec_sav', ...
                           'dec_pre_diff', 'tst_pre_diff'}))
                       fdef = rmfield(fdef, fld);
                   end
               end
           end                           
                           
           if ~isempty(dec_opts) && ~isa(dec_opts, 'CS_DecParams')
               dec_opts = CS_DecParams(dec_opts);
           end
           
           if nargin < 3
               anls_opts = [];
           elseif ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
               anls_opts = CS_AnlsParams(anls_opts);
           end
           if ~isempty(anls_opts) && isfield(fdef, 'inp_vanls')
               anls_opts.chk_ofsts = true;
           end
           
           if isfield(fdef, 'enc_vid')
               enc_out = {CodeDestFile(fdef.enc_vid)};
           else
               enc_out = {};
           end
           
           enc_info = struct('enc_opts', enc_opts, 'fdef', fdef);
                      
           %%%%%%%%%%%%%%%%%%%%%%%%%%%
           % --->Encoder START<------%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%
           
           file_start_time = tic;
           fprintf(1,'%s Processing %s\n', prefix, fdef.input);
           fprintf(1,'%s random seed=%d\n', prefix, enc_opts.random.seed);
           
           %read in the raw video
           vid_in_params = struct(...
             'ovrlp', enc_opts.blk_ovrlp,...
             'monochrom', ~enc_opts.process_color,...
             'w_type_e', enc_opts.wnd_type);
           if ~isempty(dec_opts)
             vid_in_params.w_type_d = dec_opts.wnd_type;
           end
           raw_vid_in = VidBlocksIn(enc_opts.blk_size, vid_in_params,...
             fdef.input, enc_opts.n_frames, enc_opts.start_frame-1);
           enc_info.raw_vid = raw_vid_in.vid_info;
           enc_info.raw_size = raw_vid_in.vid_size;
           if enc_info.enc_opts.n_frames  > raw_vid_in.vid_size(1,3)
               enc_info.enc_opts.setParams(struct('n_frames',...
                   raw_vid_in.vid_size(1,3)));
           end
           enc_info.vid_blocker = VidBlocker(enc_opts.blk_size, raw_vid_in);
           blk_cnt = enc_info.vid_blocker.calcBlkCnt();
           total_n_vid_blks=prod(blk_cnt);

           if enc_opts.random.rpt_temporal && ...
               ~isempty(anls_opts) && anls_opts.chk_bgrnd.mx_avg
             chk_bgrnd = cell(blk_cnt(1),blk_cnt(2));
             for k=1:numel(chk_bgrnd)
               chk_bgrnd{k} = BgrndMsrs(enc_opts.random.rpt_temporal,...
                 anls_opts.chk_bgrnd.mx_avg, anls_opts.chk_bgrnd.mn_dcd,...
                 anls_opts.chk_bgrnd.thrsh);
             end
             n_parblk = min(n_parblk, numel(chk_bgrnd));
           else
             chk_bgrnd = [];
           end
             
           if ~isempty(dec_opts)
               dec_inp = CodePipeArray(4096, 8*4096);
               enc_out = [{dec_inp} enc_out];
               if ~isempty(dec_opts) && dec_opts.init == 2
                   dec_opts.ref = fdef.input;
               end
               decoder = CSVidDecoder(fdef, anls_opts, dec_opts);
               decoder.setPrefix(prefix)
           else
               if CSVidCodec.check_coding == 2
                   fprintf('%s No check_coding while dec_opts is []!\n', prefix);
               end
               decoder = [];
           end
           
           % enc_info.doing_coding is a flag indicating that some coding is
           % done.  It is used to cause measurements encoding in the
           % parallel processing phase, if encoding is done at all.
           enc_info.doing_coding = ~isempty(enc_out);
           
           if length(enc_out) == 1
               enc_out = enc_out{1};
           end
           
           % Write preliminary information out
           base_elmnts = {enc_info.enc_opts, enc_info.raw_vid.copy()};
          
           % Determine type of quantization
           switch(enc_opts.lossless_coder)
               case enc_opts.LLC_INT
                   enc_info.enc_q_class = 'QuantMeasurementsBasic';
               case enc_opts.LLC_AC
                   enc_info.enc_q_class = 'QuantMeasurementsAC';
           end
           
           [inp_mark_blks, inp_anls, inp_vanls, enc_anls, ...
               inp_sav, inp_vsav, enc_sav] = ...
               CSVidCodec.open_analysis_output_files(enc_info);    
               
           if any(enc_opts.blk_pre_diff) 
               if isfield(fdef, 'enc_pre_diff')
                   enc_pre_diff_blks = ...
                     VidBlocksOut(fdef.enc_pre_diff, true,...
                      enc_info.vid_blocker, VidBlocker.BLK_STT_WNDW);
               else
                   enc_pre_diff_blks = ...
                       VidBlocksOut([], true, ...
                      enc_info.vid_blocker, VidBlocker.BLK_STT_WNDW);
               end
               if isfield(fdef, 'tst_pre_diff')
                   tst_pre_diff_blks = ...
                       VidBlocksOut(fdef.tst_pre_diff, true,...
                      enc_info.vid_blocker, VidBlocker.BLK_STT_WNDW);
                   enc_info.tst_pre_diff = true;
               else
                   % Unnecessary if not written out
                   enc_info.tst_pre_diff = false;
               end
               
               if ~isempty(decoder)
                   decoder.initRefPreDiff(enc_info.raw_vid.getPixelMax(),...
                       enc_info.raw_vid.createEmptyVideo(0));
               end
           end
                      
           %initialize storage for data about each block processed by
           %this codec
           blk_data_list=cell(total_n_vid_blks,1);
           qmsr_snr = cell(total_n_vid_blks,1);
           n_blk_data_enc = 0; % no. of blocks encoded
           n_blk_data_done=0;   % no. of blocks decoded

           if ~total_n_vid_blks
               nxt_blk_indx = [];
           else
               nxt_blk_indx = [1,1,1];
           end
           
           function save_motion(indx, motn, anls, sav)
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
           
           % Loop on all blocks and process them
           while ~isempty(nxt_blk_indx)
               if n_parblk > 0
                   n_blks_data = n_parblk;
               else
                   n_blks_data = 1;
               end
               blks_data = struct('blk_start_time', cell(1,n_blks_data),...
                   'enc_start_time', cell(1,n_blks_data),...
                   'blk_time', cell(1,n_blks_data),...
                   'enc_time', cell(1,n_blks_data));

               for iblk = 1:length(blks_data)
                   if isempty(nxt_blk_indx)
                       blks_data = blks_data(1:iblk-1);
                       break;
                   end
                   
                   blks_data(iblk).blk_start_time = tic;
                   blks_data(iblk).enc_start_time = tic;
                   
                   cur_blk_indx = nxt_blk_indx;
                   [nxt_blk_indx, clr_blk, enc_info] = ...
                       CSVidCodec.getNextBlock(cur_blk_indx, ...
                       raw_vid_in, enc_info);
                   blks_data(iblk).clr_blk = clr_blk;
                   blks_data(iblk).enc_info = enc_info;
                   blks_data(iblk).enc_info.rnd_seed = ...
                       enc_opts.randomSeed(cur_blk_indx, blk_cnt);
                   blks_data(iblk).enc_o = struct();
                   
                   if ~isempty(chk_bgrnd)
                     blks_data(iblk).enc_info.chk_bgrnd = ...
                       chk_bgrnd{cur_blk_indx(1), cur_blk_indx(2)};
                   else
                     blks_data(iblk).enc_info.chk_bgrnd = [];
                   end
                   
                   blks_data(iblk).enc_time = ...
                       toc(blks_data(iblk).enc_start_time);
                   blks_data(iblk).blk_time = ...
                       toc(blks_data(iblk).blk_start_time);
               end
                                  
               if n_parblk > 0
                   parfor iblk = 1:length(blks_data)
                       blks_data(iblk) = ...
                           CSVidCodec.parblk_run(prefix, blks_data(iblk),...
                           anls_opts, false);   % ~isempty(dec_opts));
                   end
               else
                   for iblk = 1:length(blks_data)
                       blks_data(iblk) = ...
                           CSVidCodec.parblk_run(prefix, blks_data(iblk),...
                           anls_opts, false);   % ~isempty(dec_opts));
                   end
               end
               
               for iblk = 1:length(blks_data)
                   blks_data(iblk).blk_start_time = tic;
                   blks_data(iblk).enc_start_time = tic;
                   
                   enc_info.cur_blk_indx = blks_data(iblk).enc_info.cur_blk_indx;
                   enc_info.vid_region = blks_data(iblk).enc_info.vid_region;
                   
                   % Save chk_brgnd
                   if isfield(blks_data(iblk).enc_o,'chk_bgrnd')
                       chk_bgrnd{enc_info.cur_blk_indx(1), ...
                         enc_info.cur_blk_indx(2)} = blks_data(iblk).enc_o.chk_bgrnd;
                       blks_data(iblk).enc_o = rmfield(blks_data(iblk).enc_o,...
                         'chk_bgrnd');
                   end
                   
                   enc_o_flds = fieldnames(blks_data(iblk).enc_o);
                   for k = 1:length(enc_o_flds)
                       fld = enc_o_flds{k};
                       enc_info.(fld) = blks_data(iblk).enc_o.(fld);
                   end
                                      
                   % Save auxiliary blocks
                   save_motion(enc_info.cur_blk_indx, enc_info.x_motion, ...
                       inp_anls, inp_sav); 
                   save_motion(enc_info.cur_blk_indx, enc_info.v_motion, ...
                       inp_vanls, inp_vsav); 
                   save_motion(enc_info.cur_blk_indx, enc_info.m_motion, ...
                       enc_anls, enc_sav); 
                   if ~isempty(enc_info.m_motion) && ~isempty(enc_anls)
                       enc_info.m_motion.setBlkInfo(...
                           enc_info.vid_blocker.getBlkInfo(enc_info.cur_blk_indx))
                       enc_anls.writeRecord(enc_info.m_motion.getCSVRecord());
                   end
                   
                   if isfield(enc_info.fdef, 'inp_mark') && ~isempty(enc_info.m_motion)
                       inp_mark_blks = enc_info.vid_region.putIntoBlkArray(...
                           enc_info.mrk_blk, inp_mark_blks);
                       err_msg = ...
                           inp_mark_blks.writeReadyFrames(enc_info.raw_vid);
                       if ischar(err_msg);
                           error('failed writing inp_mark frames: %s', err_msg);
                       end
                   end
                   
                   if any(enc_opts.blk_pre_diff)
                       enc_pre_diff_blks = ...
                           enc_info.vid_region.putIntoBlkArray(...
                           enc_info.pre_diff_blk, enc_pre_diff_blks);
                       [nfrm, pre_diff_vid] = ...
                           enc_pre_diff_blks.writeReadyFrames(enc_info.raw_vid);
                       if ischar(nfrm);
                           error('failed writing enc_pre_diff frames: %s', nfrm);
                       end
                       if nfrm && ~isempty(decoder)
                           decoder.addRefPreDiff(pre_diff_vid);
                       end
                       
                       if enc_info.tst_pre_diff
                           tst_pre_diff_blks = ...
                               enc_info.vid_region.putIntoBlkArray(...
                               enc_info.tst_pre_diff_blk, tst_pre_diff_blks);
                           err_msg = ...
                               tst_pre_diff_blks.writeReadyFrames(enc_info.raw_vid);
                           if ischar(err_msg);
                               error('failed writing tst_pre_diff frames: %s',...
                                   err_msg);
                           end
                       end
                   end
                   
                   if CSVidCodec.check_coding || ~isempty(enc_out)
                       enc_elmnts = [ base_elmnts {enc_info.vid_region, ...
                           enc_info.sens_mtrx, enc_info.quantizer, enc_info.q_msr}];
                       base_elmnts = {};
                       len_msrs = enc_info.q_msr.codeLength(enc_info);
                       [len_enc, enc_info] = CSVidCodec.encOutputElmnts(...
                           enc_elmnts, enc_info, enc_out);
                       if CSVidCodec.check_coding && ~isempty(decoder)
                           dec_src = dec_inp;
                       else
                           dec_src = enc_elmnts;
                       end
                   else
                       len_msrs = 0;
                       len_enc = 0;
                   end
                   
                   enc_info.blk_time = blks_data(iblk).enc_time + ...
                       toc(blks_data(iblk).enc_start_time);
                   
                   blk_elps_time = blks_data(iblk).blk_time +...
                       toc(blks_data(iblk).blk_start_time);
                   
                   % Save block information
                   n_blk_data_enc = n_blk_data_enc + 1;
                   blk_data_list{n_blk_data_enc} =...
                       CSVideoBlockProcessingData(enc_info.cur_blk_indx,...
                       enc_info, len_enc, len_msrs, ...
                       enc_info.sens_mtrx.getDC(enc_info.enc_cs_msrmnts));
                   blk_data_list{n_blk_data_enc}.seconds_to_process_block=...
                       blk_elps_time;
                   if CSVidCodec.report_qmsr_snr
                       q_cs_msrmnts = enc_info.quantizer.unquantize(enc_info.q_msr);
                       q_cs_msrmnts = enc_info.sens_mtrx.unsortNoClip(q_cs_msrmnts);
                       err_cs_msrmnts = q_cs_msrmnts - enc_info.enc_cs_msrmnts;
                       q_snr = 20*log10((1e-3 + norm(enc_info.enc_cs_msrmnts))/...
                           (1e-3 + norm(err_cs_msrmnts)));
                       qmsr_snr{n_blk_data_enc} = sprintf('Qmsr SNR=%4.1fdB ', q_snr);
                   else
                       qmsr_snr{n_blk_data_enc}= '';
                   end
                         
                       
                   if ~isempty(decoder)
                       % Decoding
                       
                       tests = struct();
                       if CSVidCodec.check_coding==2 && ~isempty(decoder)
                           tests.enc_info = enc_info;
                       end
                       
                       rdata = CSVidCodec.decodeBlock(decoder, dec_src, tests);
                       for k=1:length(rdata)
                         ll =n_blk_data_done + k;
                         dec_time = rdata{k}.read_dur + ...
                           rdata{k}.proc_dur + rdata{k}.write_dur;
                         enc_time = blk_data_list{ll}.seconds_to_process_block;
                         ttl_time = enc_time + dec_time;
                         blk_data_list{ll}.seconds_to_process_block=ttl_time;
                         if isfield(rdata{k},'rgn_psnr')
                           psnr_str = sprintf('PSNR= %4.1fdB ', rdata{k}.rgn_psnr);
                         else
                           psnr_str = '';
                         end
                         
                         fprintf(...
                           '%s %s %s%s %2.2f b/msr. Dur: %.3f (e=%.3f d=%.3f) sec\n',...
                           prefix, show_str(blk_data_list{ll}.blk_indx), ...
                           qmsr_snr{ll}, psnr_str, ...
                           (blk_data_list{ll}.n_msrmnts_bits/...
                           blk_data_list{ll}.n_blk_msrmnts),...
                           ttl_time, enc_time, dec_time..., dd.read_dur, dd.proc_dur, dd.write_dur, ...
                           );
                         
                       end
                       
                       n_blk_data_done = n_blk_data_done + length(rdata);
                   else
                       enc_time = blk_data_list{n_blk_data_enc}.seconds_to_process_block;
                       if ~isempty(pool)
                           enc_time = enc_time/pool.NumWorkers;
                       end
                       
                       fprintf('%s [%s].%s Dur: %.3f (av. enc %.3f) sec %.2f b/msr\n',...
                           prefix, int2str(enc_info.cur_blk_indx), ...
                           qmsr_snr{n_blk_data_enc}, blk_elps_time, enc_time, ...
                           (blk_data_list{n_blk_data_enc}.n_msrmnts_bits/...
                           blk_data_list{n_blk_data_enc}.n_blk_msrmnts));
                               
                       n_blk_data_done = n_blk_data_enc;
                   end
                                      
               end
           end  % while on blocks
           
           %calculate stats of the video blocks and store them in the
           %output
           cs_vid_io = CSVideoCodecInputOutputData(enc_opts, enc_info.raw_vid);
           cs_vid_io.calculate(blk_data_list(1:n_blk_data_enc));
           
           if ~isempty(decoder)
               %%%%%%%%%%%%%%%%%%%%%%%%%
               % --->Decoder END<------%
               %%%%%%%%%%%%%%%%%%%%%%%%%
               
               [cs_vid_io.msqr_err, cs_vid_io.psnr ]=decoder.finish();
               
               fprintf('%s msrmnts/input=%5.3f psnr=%.2f @ %.3f Mb/sec\n', prefix,...
                   cs_vid_io.ttl_msrmnts/cs_vid_io.ttl_inp_val, ...
                   cs_vid_io.psnr, cs_vid_io.bitRate()*1e-6)
           end
           
           %save the output variable both in binary format and text format
           save(fdef.mat, 'cs_vid_io', '-mat');
           cs_vid_io.writeToTextFile(fdef.txt);
           
           file_proc_time = toc(file_start_time);
           if ~isempty(decoder)  && isfield(fdef, 'output')
               fprintf('%s %f sec: %s ==> %s\n', ...
                   prefix, file_proc_time, fdef.input, fdef.output);
           elseif isfield(fdef, 'enc_vid')
               fprintf('%s %f sec: %s ==> %s\n', ...
                   prefix, file_proc_time, fdef.input, fdef.enc_vid);
           else
               fprintf('%s %f sec: %s\n', ...
                   prefix, file_proc_time, fdef.input);
           end
           %use the profile viewer
           eval(CSVidCodec.profiling_setting(...
               CSVidCodec.do_profiling).viewer);
           
       end
       
       function [inp_mark_blks, inp_anls, inp_vanls, enc_anls,...
               inp_sav, inp_vsav, enc_sav] =...
               open_analysis_output_files(enc_info)
           if isfield(enc_info.fdef, 'inp_mark')
               inp_mark_blks = ...
                   VidBlocksOut(enc_info.fdef.inp_mark, false, ...
                   enc_info.vid_blocker, VidBlocker.BLK_STT_WNDW); 
           else
               inp_mark_blks = [];
           end
           flds = {'inp_anls', 'inp_vanls', 'enc_anls';...
               'input analysis', 'input analysis as vector', ...
               'encoder measurements analysis'; [], [], [];...
               'inp_sav', 'inp_vsav', 'enc_sav'; [], [], []};
           for k=1:3
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
           inp_anls = flds{3,1};
           inp_vanls = flds{3,2};
           enc_anls = flds{3,3};
           inp_sav = flds{5,1};
           inp_vsav = flds{5,2};
           enc_sav = flds{5,3};
       end
       
       function blkd = parblk_run(prefix, blkd, anls_opts, print_done)
           blkd.blk_start_time = tic;
           blkd.enc_start_time = tic;
           
           blkd.enc_o = ...
               CSVidCodec.encodeBlock(prefix, blkd.clr_blk,  blkd.enc_info, ...
               anls_opts);
           
           blkd.enc_time = blkd.enc_time + toc(blkd.enc_start_time);
           blkd.blk_time = blkd.blk_time + toc(blkd.blk_start_time);
           
           if print_done
               fprintf('%s [%s] Encoded. %.3f sec\n',...
                   prefix, int2str(blkd.enc_info.cur_blk_indx), blkd.enc_time);
           end
                   
       end
       
       function [nxt_blk_indx, clr_blk, enc_info] = getNextBlock(cur_blk_indx, ...
               raw_vid_in, enc_info)
           
           [clr_blk, nxt_blk_indx] = ...
               raw_vid_in.getBlks(cur_blk_indx);
           
           raw_vid_in.discardFrmsBeforeBlk(cur_blk_indx);
           
           if isfield(enc_info.fdef, 'inp_mark')
               enc_info.mrk_blk = mark_blk_boundaries(clr_blk,...
                   enc_info.vid_blocker.ovrlp, enc_info.enc_opts.conv_rng, ...
                   0, enc_info.raw_vid.getPixelMax());
           else
               enc_info.mrk_blk = [];
           end
                     
           % Specify the video region (one block) we work on
           enc_info.cur_blk_indx = cur_blk_indx;
           
           zero_ext = struct('b',enc_info.enc_opts.zero_ext_b, ...
               'f', enc_info.enc_opts.zero_ext_f, 'w', enc_info.enc_opts.wrap_ext);
           enc_info.vid_region = VidRegion(...
               cur_blk_indx, enc_info.vid_blocker, zero_ext);
       end
       
       %**********************
       %  encodeBlock        *
       %**********************
       function enc_out = encodeBlock(prefix, clr_blk, enc_info, anls_opts)
         prefix = [prefix ' [' int2str(enc_info.cur_blk_indx) ']'];
         enc_opts = enc_info.enc_opts;
         vid_region = enc_info.vid_region;
         enc_out = struct();
         
         n_pxl_in_rgn = enc_info.vid_region.nPxlInRegion();
         enc_out.n_blk_msrmnts = ceil(n_pxl_in_rgn *...
           enc_opts.msrmnt_input_ratio);
         
         % create quantizer and encode it.
         enc_out.quantizer = UniformQuantizer(enc_opts.qntzr_wdth_mltplr, ...
           enc_opts.qntzr_ampl_stddev,...
           (enc_opts.qntzr_outrange_action == enc_opts.Q_SAVE),...
           sqrt(n_pxl_in_rgn));
         
         if any(enc_opts.blk_pre_diff)
           clr_blk = vid_region.multiDiffExtnd(clr_blk, ...
             enc_opts.blk_pre_diff);
           enc_out.pre_diff_blk = clr_blk;
           
           if enc_info.tst_pre_diff
             enc_out.tst_pre_diff_blk = ...
               vid_region.undo_multiDiffExtnd(...
               clr_blk, enc_opts.blk_pre_diff);
           end
         end
         
         %flatten video block into a vector
         blk_pxl_vctr = vid_region.vectorize(clr_blk);
         
         % Generate Sensing Matrix
         mt_args = enc_opts.msrmnt_mtrx.args;
         mt_args.num_columns = vid_region.ext_vec_len;
         mt_args.num_rows = enc_out.n_blk_msrmnts;
         
%          % Use this line to force square measurement matrix with DFT
%          % matrices.
%          mt_args.num_rows = pow2(nextpow2(vid_region.ext_vec_len));

         mt_args.rnd_seed = enc_info.rnd_seed;
         if ~isfield(mt_args, 'prmt')
           mt_args.prmt = struct();
         end
         mt_args.prmt.PL_mode = enc_opts.conv_mode;
         mt_args.prmt.PL_range = enc_opts.conv_rng;
         mt_args.prmt.PL_size = vid_region.ext_Cblk_size;
         mt_args.prmt.N_msrs = mt_args.num_rows;
         enc_out.sens_mtrx = SensingMatrix.construct(...
           enc_opts.msrmnt_mtrx.type, mt_args);
         
         n_no_clip = enc_out.sens_mtrx.nNoClip();
         
         % -->take compressed sensed measurements %
         ext_vec = vid_region.zeroExtnd(blk_pxl_vctr);
         enc_out.enc_cs_msrmnts = enc_out.sens_mtrx.multVec(ext_vec);
         
         if ~isempty(enc_info.chk_bgrnd)
           enc_out.chk_bgrnd = enc_info.chk_bgrnd;
           [enc_out.is_bgrnd, enc_out.bgrnd_age] = enc_out.chk_bgrnd.checkBlk(...
             enc_info.cur_blk_indx(3), enc_out.sens_mtrx, enc_out.enc_cs_msrmnts);
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
         
         % Compute next frame cross correlations of input
         if ~isempty(anls_opts) && ...
             (isfield(enc_info.fdef, 'inp_anls') || isfield(enc_info.fdef, 'inp_sav'))
           
           [nrmxcor, enc_out.x_motion] = vid_region.nextFrmXCor(...
             clr_blk, anls_opts);
           motion_found = enc_out.x_motion.motionFound();
           if motion_found
             report_cor(nrmxcor, enc_out.x_motion, 'nrmd Xcor');
           end
           
           if ~isempty(enc_info.mrk_blk)
             clr = [enc_info.raw_vid.getPixelMax(), 0];
             if CSVidCodec.report_msrmnt_xcor
               position = [0.35, 0.35];
             else
               position = [0.5, 0.5];
             end
             enc_info.mrk_blk = vid_region.drawMotionMarker(...
               enc_info.mrk_blk, position, clr, enc_out.x_motion);
           end
         else
           enc_out.x_motion = [];
           motion_found = false;
         end
         
         % Compute cross correlations of measurements
         if ~isempty(anls_opts) && (isfield(enc_info.fdef,'inp_mark') ||...
             isfield(enc_info.fdef,'inp_anls') || isfield(enc_info.fdef,'inp_sav') || ...
             isfield(enc_info.fdef,'inp_vanls') || isfield(enc_info.fdef,'inp_vsav'))
           
           if isfield(enc_info.fdef,'inp_vanls') || isfield(enc_info.fdef,'inp_vsav')
             % Computing v_motion
             [nrmmcor, enc_out.m_motion, n_sum, nrmvcor, enc_out.v_motion] =...
               next_msrs_xcor(enc_out.enc_cs_msrmnts, enc_out.sens_mtrx, ...
               vid_region, anls_opts, ext_vec);
           else
             % Not computing v_motion
             [nrmmcor, enc_out.m_motion, n_sum] =...
               next_msrs_xcor(enc_out.enc_cs_msrmnts, enc_out.sens_mtrx, ...
               vid_region, anls_opts);
             enc_out.v_motion = [];
           end
           if ~isempty(nrmmcor)
%              fprintf('%s number of terms: %d %d %d\n', prefix,...
%                min(n_sum), mean(n_sum), max(n_sum));
             if motion_found || enc_out.m_motion.motionFound()
               if anls_opts.chk_ofsts
                 report_cor(nrmvcor, enc_out.v_motion, 'nrmd Vcor');
               end
               report_cor(nrmmcor, enc_out.m_motion, 'nrmd Mcor',n_sum);
             end
             
             if ~isempty(enc_info.mrk_blk)
               clr = [0, enc_info.raw_vid.getPixelMax()];
               if ~isempty(enc_out.x_motion)
                 position = [0.65, 0.65];
               else
                 position = [0.5, 0.5];
               end
               enc_info.mrk_blk = vid_region.drawMotionMarker(...
                 enc_info.mrk_blk, position, clr, enc_out.m_motion);
             end
           end
         else
           enc_out.m_motion = [];
           enc_out.v_motion = [];
         end
         
         % Quantize measurements
         enc_q_msr = enc_out.quantizer.quantize(...
           enc_out.sens_mtrx.sortNoClip(enc_out.enc_cs_msrmnts), ...
           struct('n_no_clip', n_no_clip,...
           'mean',[], 'stdv',[]), ...
           enc_info.enc_q_class);
         
         enc_out.q_msr = enc_q_msr;
         
         % Force encoding, so it is done while parallel processing
         if (enc_info.doing_coding || CSVidCodec.check_coding)
           enc_out.enc_opts = enc_opts;
           if enc_out.q_msr.codeLength(enc_out) == 0
             warning('Unexpected zero length for measurements!')
           end
           enc_out = rmfield(enc_out, 'enc_opts');
         end
         
         enc_out.mrk_blk = enc_info.mrk_blk;
       end

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
       
       function [len_enc, enc_info] = encOutputElmnts(enc_elmnts,...
               enc_info, enc_out)
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
       %           enc_info - if present correctness of decoding is checked
       %              (only if dec_inp is not a cell array)
       %           clr_blk - source blocks (all color components) - if
       %                      present, PSNR for this block is computed.
       
       function rdata = decodeBlock(decoder, dec_inp, tests)
           decoder.setReadStartTime();
           
           while true
               if isfield(tests, 'enc_info')
                   [dec_inp, code_elmnt, decoder.info, elmnt_len] = ...
                       CSVidCodec.decInputElmnt(dec_inp, decoder.info, tests.enc_info);
               else
                   [dec_inp, code_elmnt, decoder.info, elmnt_len] = ...
                       CSVidCodec.decInputElmnt(dec_inp, decoder.info);
               end
               
               if isempty(code_elmnt)
                   break;
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
               dec_inp, dec_info, enc_info)
           
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
               
               if nargin  >= 3
                   switch class(code_elmnt)
                       case class(enc_info.enc_opts)
                           ce = enc_info.enc_opts;
                       case class(enc_info.raw_vid)
                           ce = enc_info.raw_vid;
                       case class(enc_info.quantizer)
                           ce = enc_info.quantizer;
                       case class(enc_info.vid_region)
                           ce = enc_info.vid_region;
                       case class(enc_info.sens_mtrx)
                           ce = enc_info.sens_mtrx;
                       case class(enc_info.q_msr)
                           ce = enc_info.q_msr;
                       otherwise
                           error('Unexpected code_element %s',...
                               class(code_elmnt));
                   end
                   if ~code_elmnt.isEqual(ce)
                       error('Unequal decoding of %s',...
                           class(code_elmnt));
                   end
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
           % enc_opts - a CS_EncParams objects or something which can be used
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
           %                   '<Nn>'.
           %         prefix - (0ptional) and identifying prefix to add 
           %                  before all printing. Default: '] '
           %         par_files - If true process files in parallel.
           %                     Default: false.
           % [Output]
           % cs_vid_io - a CSVideoCodecInputOutputData object which defines the input
           %             parameters and returns the output of the simulatio.
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
           case_start = tic;
           
           if nargin < 5
               proc_opts = struct();
           end
           if ~isfield(proc_opts, 'output_id')
               proc_opts.output_id = '*';
           end
           if ~isfield(proc_opts, 'case_id')
               proc_opts.case_id = '<Nn>';
           end
           if ~isfield(proc_opts, 'prefix')
               proc_opts.prefix = '] ';
           end
           if ~isfield(proc_opts, 'par_files')
               proc_opts.par_files = false;
           end
           
           if ~isa(enc_opts, 'CS_EncParams')
               enc_opts = CS_EncParams(enc_opts);
           end
           prefix = enc_opts.idStr(proc_opts.prefix);
           cs_vid_io = CSVideoCodecInputOutputData(enc_opts);
           
           if ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
               anls_opts = CS_AnlsParams(anls_opts);
           end
           
           if ~isempty(dec_opts) && ~isa(dec_opts, 'CS_DecParams')
               dec_opts = CS_DecParams(dec_opts);
           end
           
           proc_opts.case_id = enc_opts.idStr(proc_opts.case_id);
           
           % Initialize input files
           io_id = struct('dec', 0, 'case', proc_opts.case_id);

           if ischar(files_def)
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
           fprintf('%s Starting simulation case: %s\n%s   output directory=%s\n', ...
               prefix, proc_opts.case_id, prefix, enc_dir);
           
           report_opts(enc_opts, 'Encoding options', 'enc_opts', {enc_dir});
           
           if ~isempty(anls_opts)
               report_opts(anls_opts, 'Analysis options', 'anls_opts', ...
                   {enc_dir,dec_dir});
           end
           if ~isempty(dec_opts) || (~isempty(anls_opts) && (...
                   files_def.isType('dec_anls') || files_def.isType('dec_sav')))
               report_opts(dec_opts, 'Decoding options', 'dec_opts', {dec_dir});
               
               switch CSVidCodec.check_coding
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
                   fid = fopen(fullfile(dr, [fname, '.txt']), 'wt');
                   fprintf(fid,'%s\n',opts_str);
                   fclose(fid);
                   
                   opts.getJSON(fullfile(dr, [fname, '.json']));
               end
           end
           
           % Print file specificaiton info
           dirs = unique({enc_dir, dec_dir});
           for k=1:length(dirs)
               mat2json(struct('files_specs',files_def.getAllFiles()),...
                   fullfile(dirs{k},'files_specs.json'));
           end
           
           cs_vid_io.clearResults();
           rnd_seed_step = 10000;
           if proc_opts.par_files
               fdef_list = files_def.getAllFiles();
               enc_opts_fdef = cell(size(fdef_list));
               cs_vid_fdef = cell(size(fdef_list));
               random = enc_opts.random;
               for k=1:length(fdef_list)
                   enc_opts_fdef{k} = enc_opts;
                   enc_opts_fdef{k}.setParams(struct('random',random));
                   random.seed = random.seed+rnd_seed_step;
               end
               parfor k=1:length(fdef_list)
                   cs_vid_fdef{k} = CSVidCodec.run(fdef_list(k), ...
                       enc_opts_fdef{k}, anls_opts, dec_opts, sprintf('%s%d]',prefix,k));
               end
               for k=1:length(fdef_list)
                   cs_vid_io.add(cs_vid_fdef{k});
               end
           else
               indx = files_def.init_getFiles();
               k=1;
               while true
                   [fdef, indx] = files_def.getFiles(indx);
                   if isequal(fdef,[])
                       break
                   end

                   cs_vid_fdef = CSVidCodec.run(fdef, enc_opts, anls_opts, dec_opts,...
                       sprintf('%s%d]',prefix,k));
                   k = k+1;
                   cs_vid_io.add(cs_vid_fdef);
                   
                   % Changing seed for next file
                   random = enc_opts.random;
                   random.seed = random.seed+rnd_seed_step;
                   enc_opts.setParams(struct('random', random));
                   
               end
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
           % io_def - A string suitable for construction a FilesDef object, or
           %    a FilesDef object itself.
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
           %         par_files - If true process files in parallel.
           %                     Default: false.
           %         par_cases - If non-zero, number of cases to process in
           %                     parallel. Default: 0
           %
           % [Output]
           %    simul_info - a struct containing the test conditions and
           %                 the results for each case.
           
           % Set proc_opts defaults
           if nargin < 5
               proc_opts = struct();
           end
           
           if ~isfield(proc_opts, 'output_id')
               proc_opts.output_id = '*';
           end
           proc_opts.output_id = regexprep(proc_opts.output_id, '*',...
               datestr(now,'yyyymmdd_HHMM'));

           if ~isfield(proc_opts,'case_id')
               proc_opts.case_id = '<Nn>';
           end
           
           if ~isfield(proc_opts, 'prefix')
               proc_opts.prefix = '<Nn>] ';
           end
           
           if ~isfield(proc_opts, 'par_files')
               proc_opts.par_files = false;
           end
           
           if ~isfield(proc_opts, 'par_cases')
               proc_opts.par_cases = 0;
           end
           
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
           
           fprintf('full_path_of_output=%s\n',files_def.outputDir());
           
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
           
           n_cases = numel(simul_info.case_list);
           anls_opts = simul_info.anls_opts;
           dec_opts = simul_info.dec_opts;
           files_def = simul_info.files_def;
           proc_opts = simul_info.proc_opts;
           if proc_opts.par_cases
               for j_case = 1:proc_opts.par_cases:n_cases
                   end_case = min(j_case+proc_opts.par_cases-1, n_cases);
                   case_list = simul_info.case_list(j_case:end_case);
                   sml_rslt = simul_info.sml_rslt(j_case:end_case);
                   parfor i_case =1:(end_case- j_case+1)
                       if isempty(sml_rslt{i_case})
                           sml_rslt{i_case} =CSVidCodec.doSimulationCase(...
                               case_list{i_case}, anls_opts, dec_opts,...
                                files_def, proc_opts);
                       end
                   end
                   simul_info.sml_rslt(j_case:end_case) = sml_rslt;
                   
                   %save the result of the simulation so far
                   save([simul_info.files_def.outputDir() 'simul_info.mat'], '-struct',...
                       'simul_info', '-mat');
               end
           else
               for i_case = 1:n_cases
                   if isempty(simul_info.sml_rslt{i_case})
                       simul_info.sml_rslt{i_case} =CSVidCodec.doSimulationCase(...
                           simul_info.case_list{i_case}, anls_opts, dec_opts, ...
                           files_def, proc_opts);
                       
                       %save the result of the simulation si far
                       save([simul_info.files_def.outputDir() 'simul_info.mat'], ...
                          '-struct', 'simul_info', '-mat');
                   end
               end
           end
           
           %stop the stop watch
           simul_sec = toc(simulation_start);
            
           fprintf('\n   done. Start: %s. End: %s. %d sec.\n', start_date,...
               datestr(now,'yyyymmdd_HHMM'), simul_sec);
       end
       
   end %methods (Static=true)
end
