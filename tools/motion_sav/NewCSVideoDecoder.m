classdef NewCSVideoDecoder < handle
    %   NewCSVideoDecoder Perform decoding operations
    %   Detailed explanation goes here
    
    properties (Constant)
        % Control using parallel processing on cases. If 0, no parallel
        % processing is done.  Otherwise, this is the maximal number of
        % cases done in parallel (may be less if there are less cases left
        % or the pool size is smaller
        parallel_cases = 0;

        % Control using parallel processing for files
        parallel_files = false;
        
        % Control using parallel processing on blocks. If 0 no parallel
        % processing is done.  Otherwise the maximal number of
        % blocks done in parallel is the workers pool size times this value.
        parallel_blocks = 4;
%         parallel_blocks = 0;
        
    end
    
    properties
        n_parblk; % actual number of parallel blocks
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
        
        blks_done = [];
        
        % dimensions [height, width, frames]
        Ysize;  % Raw Y size
        UVsize; % Raw UV size
        
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
        
        prfx='';
        
        vid_cmpr = [];
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
        rgn_data_cnt = 0;
        rgn_blk_cnt = 0;
        max_rgn_blk_cnt = 0;
        
        y_bg_avg;    %  A vector, the mean of the background measurements
        y_bg_sumsqr;  % A vector, the sum square of the background measurements
        %y_bg_alt_avg;         % the alternative mean of the background measurements
        %y_bg_alt_std_dev;      %  the alternative standard deviation of the background measurements
        y_bg_const_var;         %  A vector, each element is the variance of the corresponding elements of the measurement vectors
        y_bg_log_l2distance;      %  A scalar, the L2 distance between the background measurement and their mean
        y_bg_sumsqr_log_l2distance;     %  A scalar, the sum square of log L2 distance
        y_bg_mean_log_l2distance;  %  A scalar, the mean of L2 distance (mu_bg)
        y_bg_var_log_l2distance;    %  A scalar, the variance of L2 distance (sigma_bg ^ 2)
        %y_bg_altvar;                 %  A scalar, sample variance of all elements of background measurements in a cube (sigma_hat ^ 2)
        weight_mean;                   % A scalar, the weight of the sample mean with a forgetting factor
        weight_var_mean;              % A scalar, the weight of the variance of weight_mean
        
        
        read_start_time;
    end
    
    methods
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
        %   slv _opts - an object of type CS_DecParams or something which
        %                can be used as an argument to construct such an object:
        %              A struct in which each field specify the property to be
        %                 changed.  An empty struct or empty array may be used if
        %                 no change to the defaults is necessary.
        %              A JSON string specifying such a struct.  If a field value is
        %                 a string beginning with an ampersand (&), the field minus
        %                 the ampersand prefix is evaluated before assignment.
        %              A string containing an '<' followed by a file name. A JSON
        %                string is read from the file and converted to a struct as
        %                above.
        %   output - A file name or file handle of the output
        function obj = NewCSVideoDecoder(files_def, anls_opts, slv_opts, output)
            obj.n_parblk = obj.parallel_blocks * matlabpool('size');
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
            
            if nargin >=4
                obj.dec_out = output;
            elseif isfield(obj.fdef, 'output')
                obj.dec_out = obj.fdef.output;
            else
                obj.dec_out = [];
            end
            
            % Setting analysis output
        end
        
        function delete(obj)
            if obj.ref_pre_diff_fid ~= -1
                fclose(obj.ref_pre_diff_fid);
            end
            if obj.err_pre_diff_fid ~= -1
                fclose(obj.err_pre_diff_fid);
            end
        end
        
        function setVidCompare(obj)
            if isfield(obj.fdef, 'input')
                obj.vid_cmpr = VidCompare(obj.info.raw_vid.getPixelMax(),...
                    obj.solver_opts.cmpr_frm_by_frm, obj.fdef.input,...
                    obj.info.raw_vid.seg_start_frame);
            end
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
            if nargin < 5
                skip = 0;
            end
            obj.pre_diff_cmpr = VidCompare(pxmx, ...
                obj.solver_opts.cmpr_frm_by_frm, ref, skip);
            
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
            obj.prfx = prefix;
        end
        
        % Decode a file
        %   Input:
        %     obj - this object
        %     prefix - prefix to prepend to messagees
        %     inp - CodeSource object to read from. If this is a string it is
        %           assumed to be an input file name
        %     ref_video - (optional) reference video source to compare with
        %                 the decoded video and compute PSNR. ref_video can
        %                 be one of:
        %                 * A  a RawVidInfo object describing a source file
        %                 * A file name (JSON description of the file)
        %                 * A struct with fields 'fname' for the file name
        %                   and 'skip' for the number of frames to skip.        %               
        %   Output:
        %     cs_vid_io - a CSVideoCodecInputOutputData object which defines
        %             the input parameters and returns the output of the simulation.
       
        function cs_vid_io = run(obj, prefix, inp)
            obj.setReadStartTime();
            
            if ischar(inp)
                input = CodeSourceFile(inp);
            else
                input = inp;
            end
            
            n_rgn_processed = 0;
            
            while isempty(obj.blks_done) || ~all(obj.blks_done(:));
                [code_elmnt, len, obj.info] = ...
                    CodeElement.readElement(obj.info, input);
                if ischar(code_elmnt)
                    exc = MException('%s NewCSVideoDecoder:run',...
                        ['Error in CodeElement:readElement(): '...
                        prefix, code_elmnt]);
                    throw(exc);
                elseif isscalar(code_elmnt) && code_elmnt == -1
                    break;
                end
                
                obj.total_bytes = obj.total_bytes + len;
                
                rdata = obj.decodeItem(code_elmnt, len);
                
                if ischar(rdata) 
                    error('%s %s', prefix, cs_msrs);
                end
                
                for k=1:length(rdata)
                    dec_data = rdata{k};
                    
                    n_rgn_processed = n_rgn_processed + 1;
                    bdata = CSVideoBlockProcessingData(...
                        dec_data.info.vid_region.blk_indx(1,:), ...
                        dec_data.info,...
                        dec_data.blk_len, dec_data.msrs_len, obj.dc_val);
                    obj.cs_blk_data_cnt = obj.cs_blk_data_cnt + 1;
                    obj.cs_blk_data_list{obj.cs_blk_data_cnt} = bdata;
                    
                    ttl_dur = dec_data.read_dur + dec_data.proc_dur +...
                        dec_data.write_dur;
                    
                    fprintf('%s blk %s done. R+P+W = %f +%f +%f = %f sec.\n', ...
                        dec_data.prfx,...
                        int2str(dec_data.info.vid_region.blk_indx(1,:)),...
                        dec_data.read_dur, dec_data.proc_dur, ...
                        dec_data.write_dur, ttl_dur);
                    
                    obj.total_dur = obj.total_dur + ttl_dur;
                end
            end
            
            if nargout > 0
                cs_vid_io = CSVideoCodecInputOutputData(obj.info.enc_opts,...
                    obj.info.raw_vid);
                cs_vid_io.calculate(obj.cs_blk_data_list(1:obj.cs_blk_data_cnt));
                
                [cs_vid_io.msqr_err, cs_vid_io.psnr] = ...
                    obj.finish();
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
            output = CodeDestFile(vid_out);
            obj.rgn_code_len = 0;
            n_vid_blks_processed = 0;
            out_info = struct();
            
            elmnts = cell(10,1);
            n_elmnts = 0;
            while true
               [code_elmnt, ~, obj.info] = ...
                    CodeElement.readElement(obj.info, input);
                if ischar(code_elmnt)
                    error('%s NewCSVideoDecoder:run',...
                        ['Error in CodeElement:readElement(): '...
                        prefix, code_elmnt]);
                elseif isscalar(code_elmnt) && code_elmnt == -1
                    break;
                end
                
                n_elmnts = n_elmnts+1;
                elmnts{n_elmnts} = code_elmnt;
                
                if isa(code_elmnt, 'QuantMeasurements')
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
                    bdata = CSVideoBlockProcessingData(...
                        out_info.vid_region.blk_indx(1,:), out_info,...
                        obj.rgn_code_len);
                    obj.cs_blk_data_cnt = obj.cs_blk_data_cnt + 1;
                    obj.cs_blk_data_list{obj.cs_blk_data_cnt} = bdata;
                    
                    fprintf('%s blk %s done.\n', prefix,...
                        int2str(out_info.vid_region.blk_indx(1,:)));
                    
                    obj.rgn_code_len = 0;

                elseif isa(code_elmnt, 'SensingMatrix')
                    obj.info.sens_mtrx = code_elmnt;
                elseif isa(code_elmnt, 'UniformQuantizer')
                    obj.info.quantizer = code_elmnt;
                    out_info.quantizer = code_elmnt;
                    obj.info.enc_opts.qntzr_wdth_mltplr = code_elmnt.q_wdth_mltplr;
                    obj.info.enc_opts.qntzr_ampl_stddev = code_elmnt.q_ampl_mltplr;
                elseif isa(code_elmnt, 'VidRegion')
                    obj.info.vid_region = code_elmnt;
                    out_info.vid_region = code_elmnt;
                elseif isa(code_elmnt, 'CS_EncParams')
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

                    if ~isempty(obj.solver_opts) && isempty(obj.vid_cmpr)
                        obj.setVidCompare();
                    end
                    
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
        %   item - Item to decode
        % Output:
        %   done - normally 0. 1 if a video region decoding is completed.
        %          -1 if an unexpected
        %          item was read.
        %   
        function rdata = decodeItem(obj, item, len)
            rdata = [];
            
            obj.rgn_code_len = obj.rgn_code_len + len;
            
            if isa(item, 'QuantMeasurements')
                obj.info.q_msr = item;
                
                %bx = obj.info.vid_region.blk_indx(1,:);
                %if bx(1)~=5 || bx(2)~=5
                 %   return;
                %end
                dec_data = struct(...
                    'prfx', obj.prfx,...
                    'info', obj.info,...
                    'anls_opts', obj.anls_opts,...
                    'solver_opts', obj.solver_opts,...
                    'q_msrs', item,...
                    'ref_blks', [],...
                    'dc_val',[],...
                    'blk_motion',[],...
                    'pre_diff_blks', ~isempty(obj.dec_pre_diff_blks),...
                    'dec_anls', ~isempty(obj.dec_anls),...
                    'dec_sav', ~isempty(obj.dec_sav),...
                    'dec_blks', ~isempty(obj.output_blks),...
                    'dec_slct_mrk_blks', ~isempty(obj.dec_slct_mark_blks),...
                    'dec_mrk_blks', ~isempty(obj.dec_mark_blks),...
                    'ref_mrk_blks', ~isempty(obj.ref_mark_blks),...
                    'msrs_len', len,...
                    'blk_len',obj.rgn_code_len,...
                    'read_dur', 0,...
                    'proc_dur', 0,...
                    'write_dur', 0 ...
                    );
                
                obj.rgn_code_len = 0;
                
                dec_data.read_dur = toc(obj.read_start_time);
                
                if (~isempty(obj.anls_opts) && ~isempty(obj.ref_mark_blks)) ||...
                        (~isempty(obj.solver_opts) && obj.solver_opts.init==-1)
                    dec_data.ref_blks = obj.getRefRegion();
                    obj.ref_rgn_data = [];
                end
                
                obj.rgn_blk_cnt = obj.rgn_blk_cnt + ...
                    size(obj.info.vid_region.blk_indx,1);
                obj.rgn_data_cnt = obj.rgn_data_cnt + 1;
                obj.rgn_data{obj.rgn_data_cnt} = dec_data;
                
%                 fprintf('%s blk %s read. %.3f sec\n', obj.prfx,...
%                     int2str(dec_data.info.vid_region.blk_indx(1,:)), dec_data.read_dur);
% 
                if obj.rgn_blk_cnt >= obj.max_rgn_blk_cnt % do parallel processing
                    % Calclulate
                    rdata = obj.rgn_data(1:obj.rgn_data_cnt);
                    if obj.n_parblk
                        parfor k=1:obj.rgn_data_cnt
                            rdata{k} = ...
                                NewCSVideoDecoder.get_cs_msrsDecData(rdata{k});
                        end
                    else
                        for k=1:obj.rgn_data_cnt
                            rdata{k} = ...
                                NewCSVideoDecoder.get_cs_msrsDecData(rdata{k});
                        end
                    end
                    
                    
                  
                    
                    for k=1:obj.rgn_data_cnt
                        rdata{k} = compActivitiyStatistic(obj, rdata{k});
                    end
                        
                    if obj.n_parblk
                        parfor k=1:obj.rgn_data_cnt
                            rdata{k} = ...
                                NewCSVideoDecoder.calculateDecData(rdata{k});
                        end
                    else
                        for k=1:obj.rgn_data_cnt
                            rdata{k} = ...
                                NewCSVideoDecoder.calculateDecData(rdata{k});
                        end
                    end
                    
                    % Writing out
                    for k=1:obj.rgn_data_cnt
                        rdata{k} = obj.writeDecData(rdata{k});
                    end
                    
                   obj.rgn_blk_cnt = 0; 
                   obj.rgn_data_cnt = 0;
                end
                
                obj.setReadStartTime();
                
            elseif isa(item, 'SensingMatrix')
                obj.info.sens_mtrx = obj.info.vid_region.getExtndMtrx(item);
            elseif isa(item, 'UniformQuantizer')
                obj.info.quantizer = item;
                obj.info.enc_opts.setParams(struct(...
                'qntzr_wdth_mltplr', item.q_wdth_mltplr,...
                'qntzr_ampl_stddev', item.q_ampl_mltplr));
            elseif isa(item, 'VidRegion')
                obj.info.vid_region = item;
            elseif isa(item, 'CS_EncParams')
                obj.info.enc_opts = item;
                obj.info.enc_opts.setParams(struct(...
                    'dec_opts', obj.solver_opts));
                obj.info.Yblk_size = item.blk_size;
                obj.initBlocker();
            elseif isa(item, 'RawVidInfo')
                obj.info.raw_vid = item;
                                
                obj.initBlocker();
                
                if isfield(obj.solver_opts, 'ref') && ischar(obj.solver_opts.ref)
                    % Read initial video
                    [~, ref_vid_data, ref_vid_err_msg] = ...
                        read_raw_video(obj.solver_opts.ref,...
                        obj.info.enc_opts.n_frames,...
                        obj.info.raw_vid.seg_start_frame+1);
                    if ~isempty(ref_vid_err_msg)
                        error('-- Failed to read in file ''%s'' with error ''%s''.\n',...
                            obj.solver_opts.ref, raw_vid_err_msg);
                    else
                        obj.ref_blk_data = obj.info.vid_blocker.getBlk(...
                            ref_vid_data, []);
                    end
                    
                end

                if ~isempty(obj.solver_opts) && isempty(obj.vid_cmpr)
                    obj.setVidCompare();
                end
                    
            else
                rdata = ['Unknown code element ', class(item)];
            end
            
        end
        
        % Compute statistics of background measurements
        
          function dec_data = compActivitiyStatistic(obj, dec_data)
            
            fgt_factor = 0.9; % forgetting factor
            init_avg_cnt=3;  % no. of blocks for initial averaging
            b_indx = dec_data.info.vid_region.blk_indx;
            if isempty(obj.y_bg_sumsqr{b_indx(1), b_indx(2)})
                obj.y_bg_sumsqr{b_indx(1),b_indx(2)}=...
                    dec_data.cs_msrs(2:end).^2;
                obj.y_bg_avg{b_indx(1),b_indx(2)}=dec_data.cs_msrs(2:end);
                %obj.y_bg_altvar{b_indx(1),b_indx(2)}= 0;  % A scalar, sample variance of all elements of background measurements in a cube; initial value is 0.
                obj.y_bg_const_var{b_indx(1),b_indx(2)}= zeros(length(dec_data.cs_msrs(2:end)),1); % A vector of zeros
                
                
                % initialize the weight of the forgetting factor sample mean and the weight of its variance
                obj.weight_mean{b_indx(1),b_indx(2)} = (1 - fgt_factor^init_avg_cnt)/(1-fgt_factor);
                obj.weight_var_mean{b_indx(1),b_indx(2)} = ((1-fgt_factor)*(1+fgt_factor^init_avg_cnt))/((1-fgt_factor^init_avg_cnt)*...
                    (1+fgt_factor));
                
                % initialize y_bg_log_l2distance and related statistics
                obj.y_bg_log_l2distance{b_indx(1),b_indx(2)}=0;
                
                obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}=0;
                obj.y_bg_sumsqr_log_l2distance{b_indx(1),b_indx(2)}=0;
                obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)}=0;
                
                % initialize dec_data.(statistics)
                
                dec_data.y_bg_log_l2distance = obj.y_bg_log_l2distance{b_indx(1),b_indx(2)};
                
                dec_data.y_bg_mean_log_l2distance = obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_var_log_l2distance = obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_const_var =  obj.y_bg_const_var{b_indx(1),b_indx(2)};
                
                
            elseif b_indx(3)<=init_avg_cnt
                % pass the statistic values to dec_data
                
                dec_data.y_bg_log_l2distance = obj.y_bg_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_mean_log_l2distance = obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_var_log_l2distance = obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_const_var =  obj.y_bg_const_var{b_indx(1),b_indx(2)};
                
                
                % compute the statistics of background measurements
                obj.y_bg_sumsqr{b_indx(1),b_indx(2)}=...
                    obj.y_bg_sumsqr{b_indx(1),b_indx(2)}+...
                    dec_data.cs_msrs(2:end).^2;
                
                obj.y_bg_avg{b_indx(1),b_indx(2)}=((b_indx(3)-1)/b_indx(3))*...
                    obj.y_bg_avg{b_indx(1),b_indx(2)}+(1/b_indx(3))*dec_data.cs_msrs(2:end);
                
                obj.y_bg_const_var{b_indx(1),b_indx(2)}=(1/(b_indx(3)-2))*...
                    (obj.y_bg_sumsqr{b_indx(1),b_indx(2)}-...
                    (b_indx(3)-1)* obj.y_bg_avg{b_indx(1),b_indx(2)}.^2);
                
                
                
               % obj.y_bg_altvar{b_indx(1),b_indx(2)}= (1/((b_indx(3)*length(dec_data.cs_msrs))-1))...
                %    *(sum(obj.y_bg_sumsqr{b_indx(1),b_indx(2)})-b_indx(3)*...
                 %   dot(obj.y_bg_avg{b_indx(1),b_indx(2)},obj.y_bg_avg{b_indx(1),b_indx(2)}));
                
                % compute y_bg_log_l2distance and related statistics
                
                obj.y_bg_log_l2distance{b_indx(1),b_indx(2)}=log(dot(dec_data.cs_msrs(2:end)-...
                    obj.y_bg_avg{b_indx(1),b_indx(2)},dec_data.cs_msrs(2:end)-...
                    obj.y_bg_avg{b_indx(1),b_indx(2)}));
                
                obj.y_bg_sumsqr_log_l2distance{b_indx(1),b_indx(2)}= ...
                    obj.y_bg_sumsqr_log_l2distance{b_indx(1),b_indx(2)}+...
                    obj.y_bg_log_l2distance{b_indx(1),b_indx(2)}^2;
                obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}=((b_indx(3)-2)/(b_indx(3)-1))*...
                    obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}+...
                    (1/(b_indx(3)-1))*obj.y_bg_log_l2distance{b_indx(1),b_indx(2)};
                obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)}=(1/(b_indx(3)-2))*...
                    (obj.y_bg_sumsqr_log_l2distance{b_indx(1),b_indx(2)}-...
                    (b_indx(3)-1)*obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}^2);
                
                fprintf('L2_distance=%f\n', ...
                    obj.y_bg_log_l2distance{b_indx(1),b_indx(2)});
                fprintf('miu_bg=%f\n',...
                    obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)});
                fprintf('sigma_bg^2=%f\n',...
                    obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)});
                fprintf('sigma_hat^2=%f\n',...
                    mean(obj.y_bg_const_var{b_indx(1),b_indx(2)}));
                fprintf('block_index=%d\n',...
                    dec_data.info.vid_region.blk_indx);
            else %compute the forgetting statistics
                
                % pass the statistic values to dec_data
                bg_diff = dec_data.cs_msrs(2:end)-obj.y_bg_avg{b_indx(1),b_indx(2)};
                dec_data.y_bg_log_l2distance = log(dot(bg_diff, bg_diff));
                %dec_data.y_bg_log_l2distance = obj.y_bg_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_mean_log_l2distance = obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)};
                dec_data.y_bg_var_log_l2distance = obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)};
                 dec_data.y_bg_const_var =  obj.y_bg_const_var{b_indx(1),b_indx(2)};
                
                % compute previous weight of the forgetting factor sample
                % variance
                weight_var_old = obj.weight_mean{b_indx(1),b_indx(2)}*(1-obj.weight_var_mean{b_indx(1),b_indx(2)});
                
                % update the weight of the forgetting factor sample mean and the weight of its variance
                
                obj.weight_mean{b_indx(1),b_indx(2)} = fgt_factor*obj.weight_mean{b_indx(1),b_indx(2)} + 1;
                obj.weight_var_mean{b_indx(1),b_indx(2)} = ((obj.weight_mean{b_indx(1),b_indx(2)}-1)/obj.weight_mean{b_indx(1),b_indx(2)})^2*...
                    obj.weight_var_mean{b_indx(1),b_indx(2)}+(1/obj.weight_mean{b_indx(1),b_indx(2)})^2;
                
                % compute current weight of the forgetting factor sample
                % variance
                weight_var_new = obj.weight_mean{b_indx(1),b_indx(2)}*(1-obj.weight_var_mean{b_indx(1),b_indx(2)});
                
                
                
                % compute the statistics of background measurements
                obj.y_bg_sumsqr{b_indx(1),b_indx(2)}=...
                    ((init_avg_cnt- 1)/init_avg_cnt)*(obj.y_bg_sumsqr{b_indx(1),b_indx(2)})+...
                    dec_data.cs_msrs(2:end).^2;
                
                
                 obj.y_bg_const_var{b_indx(1),b_indx(2)}= (1/weight_var_new)*(fgt_factor*weight_var_old*...
                    obj.y_bg_const_var{b_indx(1),b_indx(2)} + ((obj.weight_mean{b_indx(1),b_indx(2)}-1)/obj.weight_mean{b_indx(1),b_indx(2)})*...
                    (obj.y_bg_avg{b_indx(1),b_indx(2)}-dec_data.cs_msrs(2:end)).^2);
              
                
                obj.y_bg_avg{b_indx(1),b_indx(2)}=(1-1/obj.weight_mean{b_indx(1),b_indx(2)})*...
                    obj.y_bg_avg{b_indx(1),b_indx(2)}+(1/obj.weight_mean{b_indx(1),b_indx(2)})*dec_data.cs_msrs(2:end);
                
             %   obj.y_bg_altvar{b_indx(1),b_indx(2)}= (1/((init_avg_cnt*length(dec_data.cs_msrs))-1))...
              %      *(sum(obj.y_bg_sumsqr{b_indx(1),b_indx(2)})-init_avg_cnt*...
               %     dot(obj.y_bg_avg{b_indx(1),b_indx(2)},obj.y_bg_avg{b_indx(1),b_indx(2)}));
                
                % compute y_bg_log_l2distance and related statistics
                obj.y_bg_log_l2distance{b_indx(1),b_indx(2)}=log(dot(dec_data.cs_msrs(2:end)-...
                    obj.y_bg_avg{b_indx(1),b_indx(2)},dec_data.cs_msrs(2:end)-...
                    obj.y_bg_avg{b_indx(1),b_indx(2)}));
                
                obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)}=(1/weight_var_new)*(fgt_factor*weight_var_old*...
                    obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)} + ((obj.weight_mean{b_indx(1),b_indx(2)}-1)/obj.weight_mean{b_indx(1),b_indx(2)})*...
                    (obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}-obj.y_bg_log_l2distance{b_indx(1),b_indx(2)})^2);
                
                obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)} = (1-1/obj.weight_mean{b_indx(1),b_indx(2)})*...
                    obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)} + (1/obj.weight_mean{b_indx(1),b_indx(2)})* ...
                    obj.y_bg_log_l2distance{b_indx(1),b_indx(2)};
                
                
                obj.y_bg_sumsqr_log_l2distance{b_indx(1),b_indx(2)}= ...
                    ((init_avg_cnt- 1)/init_avg_cnt)*obj.y_bg_sumsqr_log_l2distance{b_indx(1),b_indx(2)}+...
                    obj.y_bg_log_l2distance{b_indx(1),b_indx(2)}^2;
                %  obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}=((init_avg_cnt-1)/init_avg_cnt)*...
                %     obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)}+...
                %     (1/init_avg_cnt)*obj.y_bg_log_l2distance{b_indx(1),b_indx(2)};
                
                fprintf('L2_distance=%f\n', ...
                    obj.y_bg_log_l2distance{b_indx(1),b_indx(2)});
                fprintf('miu_bg=%f\n',...
                    obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)});
                fprintf('sigma_bg^2=%f\n',...
                    obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)});
                fprintf('sigma_hat^2=%f\n',...
                     mean(obj.y_bg_const_var{b_indx(1),b_indx(2)}));
                fprintf('block_index=%d\n',...
                    dec_data.info.vid_region.blk_indx);
            end
            
            %pass the statistical values to dec_data
            %  bg_diff = dec_data.cs_msrs-obj.y_bg_avg{b_indx(1),b_indx(2)};
            % dec_data.y_bg_log_l2distance = log(dot(bg_diff, bg_diff));
            
            % dec_data.y_bg_mean_log_l2distance = obj.y_bg_mean_log_l2distance{b_indx(1),b_indx(2)};
            %dec_data.y_bg_var_log_l2distance = obj.y_bg_var_log_l2distance{b_indx(1),b_indx(2)};
            %dec_data.y_bg_altvar = obj.y_bg_altvar{b_indx(1),b_indx(2)};
        end
        
        function [mse, psnr] = finish(obj)
             dec_finish_start = tic;
                        
            %%%%%%%%%%%%%%%%%%%%%%%%%
            %--> unblock video      %
            %%%%%%%%%%%%%%%%%%%%%%%%%

            if any(obj.info.enc_opts.blk_pre_diff) && ~isempty(obj.pre_diff_cmpr)
                pre_diff_psnr = obj.pre_diff_cmpr.getPSNR();
                fprintf('%s pre diff PSNR: %f\n', obj.prfx, pre_diff_psnr);
            end
          
            dur = toc(dec_finish_start);
            obj.total_dur = obj.total_dur + dur;
            
            % If necessary run comparisons
            if ~isempty(obj.vid_cmpr) && obj.vid_cmpr.sqr_err.n_pnt
                [psnr, mse] = obj.vid_cmpr.getPSNR();
            else
                psnr = 0; mse=0;
            end
        end
    end
    
    methods (Access=private)
        
        function dec_data = writeDecData(obj, dec_data)
            start_time = tic;
            
            obj.dc_val = dec_data.dc_val;
            
            if ~isempty(dec_data.pre_diff_blks)
                obj.writePreDiff(dec_data.pre_diff_blks);
            end
            
            % Write analysis into CSV file
            if dec_data.dec_anls || dec_data.dec_sav
                for k = 1:dec_data.info.vid_region.n_blk
                    b_indx = dec_data.info.vid_region.blk_indx(k,:);
                    dec_data.blk_motion.setBlkInfo(...
                        dec_data.info.vid_blocker.getBlkInfo(b_indx));
                    if dec_data.dec_anls
                        obj.dec_anls.writeRecord(...
                            dec_data.blk_motion.getCSVRecord());
                    end
                    if dec_data.dec_sav
                        obj.dec_sav.setBlkRecord(dec_data.blk_motion);
                    end
                end
            end
            
            % Write decoded files
            if ~isempty(dec_data.dec_blks)
                obj.output_blks = dec_data.info.vid_region.putIntoBlkArray(...
                    dec_data.dec_blks, obj.output_blks);
                [nfrm, vid] = obj.output_blks.writeReadyFrames(dec_data.info.raw_vid);
                if ischar(nfrm)
                    error('Failed writing decoded frames (%s)', nfrm);
                end
                if nfrm > 0 && ~isempty(obj.vid_cmpr)
                    obj.vid_cmpr.update(vid, obj.prfx);
                end
            end
            
            if ~isempty(dec_data.dec_slct_mrk_blks)
                obj.dec_slct_mark_blks = dec_data.info.vid_region.putIntoBlkArray(...
                    dec_data.dec_slct_mrk_blks, obj.dec_slct_mark_blks);
                err_msg = ...
                    obj.dec_slct_mark_blks.writeReadyFrames(dec_data.info.raw_vid);
                if ischar(err_msg);
                    error('failed writing dec_slct_mark frames: %s', err_msg);
                end
            end
            
            if ~isempty(dec_data.dec_mrk_blks) && ~isempty(obj.dec_mark_blks)
                obj.dec_mark_blks = dec_data.info.vid_region.putIntoBlkArray(...
                    dec_data.dec_mrk_blks, obj.dec_mark_blks);
                err_msg = ...
                    obj.dec_mark_blks.writeReadyFrames(dec_data.info.raw_vid);
                if ischar(err_msg);
                    error('failed writing dec_mark frames: %s', err_msg);
                end
            end
            
            if ~isempty(dec_data.ref_blks) && ~isempty(obj.ref_mark_blks)
                obj.ref_mark_blks = dec_data.info.vid_region.putIntoBlkArray(...
                    dec_data.ref_blks, obj.ref_mark_blks);
                err_msg = ...
                    obj.ref_mark_blks.writeReadyFrames(dec_data.info.raw_vid);
                if ischar(err_msg);
                    error('failed writing ref_mark frames: %s', err_msg);
                end
            end
            obj.blks_done = dec_data.info.vid_region.markDone(obj.blks_done);
            
            dec_data.write_dur = toc(start_time);
        end
        
        function ref_rgn = getRefRegion(obj)
            if isempty(obj.ref_rgn_data) && isfield(obj.fdef, 'input')
                if isempty(obj.ref_rgn_src)
                    spec = obj.info.enc_opts;
                    obj.ref_rgn_src = VidBlocksIn(spec.blk_size, ...
                        struct(...
                        'ovrlp', spec.blk_ovrlp,...
                        'monochrom', ~spec.process_color,...
                        'w_blk', spec.wnd_blk_enc,...
                        'w_type', spec.wnd_type),...
                        obj.fdef.input, spec.n_frames, spec.start_frame-1);
                end
                
                obj.ref_rgn_data = cell(size(obj.info.vid_region.blk_indx))';
                for k = 1:obj.info.vid_region.n_blk
                    b_indx = obj.info.vid_region.blk_indx(k,:);
                    blk = obj.ref_rgn_src.getBlks(b_indx);
                    if isempty(blk)
                        error('%s Ref block index [%s] is empty', obj.prfx,...
                            int2str(b_indx));
                    end
                    obj.ref_rgn_data(:,k) = blk;
                end
                obj.ref_rgn_src.discardFrmsBeforeBlk(...
                    obj.info.vid_region.blk_indx(1,:));
            end
            ref_rgn = obj.ref_rgn_data;
        end
        
        function initBlocker(obj)
            if ~isfield(obj.info,'blocker') && isfield(obj.info, 'raw_vid')...
                    && ~isempty(obj.info.enc_opts)
                
                %calculate the dimensions of the read in video
                obj.Ysize=[...
                    obj.info.raw_vid.height,...
                    obj.info.raw_vid.width,...
                    obj.info.enc_opts.n_frames];

                if obj.info.raw_vid.UVpresent
                    obj.UVsize = [obj.info.raw_vid.UVheight,...
                        obj.info.raw_vid.UVwidth,...
                        obj.info.enc_opts.n_frames];
                    
                    obj.info.raw_size = [obj.Ysize; obj.UVsize; obj.UVsize];
                else
                    obj.info.UVblk_size= [0,0,0];
                    obj.info.raw_size = obj.Ysize;
                end
                raw_size = obj.info.raw_size;
                if ~obj.info.enc_opts.process_color
                    raw_size = raw_size(1,:);
                end
                
                obj.info.vid_blocker = VidBlocker(obj.info.Yblk_size, struct(...
                    'vid_size', raw_size,...
                    'ovrlp', obj.info.enc_opts.blk_ovrlp,...
                    'zext', [obj.info.enc_opts.zero_ext_b; ...
                    obj.info.enc_opts.zero_ext_f],...
                    'w_blk', obj.info.enc_opts.wnd_blk_enc,...
                    'w_type', obj.info.enc_opts.wnd_type,...
                    'fps', obj.info.raw_vid.fps));
                
                
                % Calculate number of blocks
                blk_cnt = obj.info.vid_blocker.calcBlkCnt();
                
                % create region storage area
                obj.max_rgn_blk_cnt = min(blk_cnt(1)*blk_cnt(2),...
                    max(1,obj.n_parblk));
                obj.rgn_data = cell(1,obj.max_rgn_blk_cnt);

                % allocate space for background
                   obj.y_bg_avg = cell(blk_cnt(1),blk_cnt(2));
                obj.y_bg_sumsqr = cell(blk_cnt(1),blk_cnt(2));
                %obj.y_bg_alt_avg = cell(blk_cnt(1),blk_cnt(2));
                %obj.y_bg_alt_std_dev = cell(blk_cnt(1),blk_cnt(2));
                obj.y_bg_const_var = cell(blk_cnt(1),blk_cnt(2));
                obj.y_bg_log_l2distance = cell(blk_cnt(1),blk_cnt(2));
                obj.y_bg_mean_log_l2distance = cell(blk_cnt(1),blk_cnt(2));
                obj.y_bg_var_log_l2distance = cell(blk_cnt(1),blk_cnt(2));
                obj.y_bg_sumsqr_log_l2distance = cell(blk_cnt(1),blk_cnt(2));
               % obj.y_bg_altvar = cell(blk_cnt(1),blk_cnt(2));
               % obj.y_l2dh = zeros(blk_cnt(1), blk_cnt(2), blk_cnt(3));
                %obj.y_l2mu = zeros(blk_cnt(1), blk_cnt(2), blk_cnt(3));
                %obj.y_l2sg2 = zeros(blk_cnt(1), blk_cnt(2), blk_cnt(3));
                obj.weight_mean = cell(blk_cnt(1),blk_cnt(2));
                obj.weight_var_mean = cell(blk_cnt(1),blk_cnt(2));
                
                %initialize storage for recovered video block
                
                obj.blks_done = zeros(blk_cnt);
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
                        obj.ref_mark_blks = VidBlocksOut(obj.fdef.ref_mark, false,...
                            obj.info.vid_blocker);
                    end
                    
                    % open output files requiring both analysis and
                    % reconstruction
                    if ~ isempty(obj.solver_opts)
                        if isfield(obj.fdef, 'dec_mark')
                            obj.dec_mark_blks = VidBlocksOut(obj.fdef.dec_mark, false,...
                                obj.info.vid_blocker);
                        end
                        if isfield(obj.fdef, 'dec_slct_mark')
                            obj.dec_slct_mark_blks = VidBlocksOut(...
                                obj.fdef.dec_slct_mark, false, obj.info.vid_blocker);
                        end
                    end
                end
                % Open output files requiring reconstruction
                if ~isempty(obj.solver_opts)
                    if ~isempty(obj.dec_out)
                        obj.output_blks = VidBlocksOut(obj.dec_out,...
                            false, obj.info.vid_blocker);
                    end
                    
                    if any(obj.info.enc_opts.blk_pre_diff) &&...
                            (isfield(obj.fdef, 'dec_pre_diff') ||...
                            isfield(obj.fdef, 'err_pre_diff') ||...
                            isfield(obj.fdef, 'dec_ref_diff'))
                        
                        if isfield(obj.fdef, 'dec_pre_diff')
                            obj.dec_pre_diff_blks = VidBlocksOut(...
                                obj.fdef.dec_pre_diff, true, obj.info.vid_blocker);
                        else
                            obj.dec_pre_diff_blks = VidBlocksOut(...
                                [], true, obj.info.vid_blocker);
                        end
                    end
                end
                
            end
        end
        
        function writePreDiff(obj,pre_diff_blks)
            obj.dec_pre_diff_blks = ...
                obj.info.vid_region.putIntoBlkArray(...
                pre_diff_blks, obj.dec_pre_diff_blks);
            [nfrm, dec_pre_diff] = ...
                obj.dec_pre_diff_blks.writeReadyFrames(obj.info.raw_vid);
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
                
                for kk=1:length(out_vid)
                    out_vid{kk} = obj.info.raw_vid.convertValsToPxls(...
                        0.5*(out_vid{kk}+obj.info.raw_vid.getPixelMax()+1));
                end
                
                err_msg = write_raw_video(fid, out_vid);
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
                   [recovered_input, ~] = TVAL3_CVS_D2(handle_sens_mtrx, ...
                       double(compressed_sensed_measurements), bi(1),bi(2),bi(3),opts);
           end
           
           recovered_input = sparser.vid_region.pixelize(recovered_input);
           
       end
       
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
       %     params   - (optional) a struct which contains optioanl
       %                parameters. Its fields may be:
       %           prefix - prefix of printed lines
       %           case_dir - the case directory name (forces this name)
       %           io_params - A struct of the type used as a second
       %                       argumnt to the constructor of FilesDef
      function cs_vid_io = doSimulationCase(anls_opts, dec_opts, files_def,  params)
           case_start = tic;
           
           if nargin < 3
               params = struct();
           end
           
           if ~isfield(params, 'prefix')
               prefix = '';
           else
               prefix = params.prefix;
           end

           % Write time stamp into dec_id if there is a '*' there.
           if isfield(params, 'io_params') && ...\
                   isfield(params.io_params,'identifier') &&...
                   isfield(params.io_params.identifier,'dec')
               params.io_params.identifier.dec = regexprep(...
                   params.io_params.identifier.dec, '*',...
                   datestr(now,'yyyymmdd_HHMM'),'once');
           end

           if isa(files_def, 'FilesDef')
               if isfield(params, 'io_params') && isfield(params.io_params,...
                       'identifier')
                   files_def.specifyDirs(params.io_params.identifier);
               end
           elseif isfield(params, 'io_params')
               files_def = FilesDef(files_def, params.io_params);
           else
               files_def = FilesDef(files_def);
           end
           if isfield(params, 'case_dir')
               files_def.setDirs(struct('case', params.case_dir));
           end
               
           fprintf('%s Starting simulation case on: %s\n%s output directory=%s \n', ...
               prefix, files_def.caseDir(),  prefix, files_def.getDecoderDir());

           files_def.makeDirs();
           dec_dir = files_def.getDecoderDir();
           
           % Print file specification info
           mat2json(struct('files_specs',files_def.getAllFiles()),...
               fullfile(dec_dir,'files_specs.json'));

           
           if ~isempty(anls_opts) 
               if ~isa(anls_opts,'CS_AnlsParams')
                   anls_opts = CS_AnlsParams(anls_opts);
               end
               fprintf('%s\n',anls_opts.describeParams(prefix));
               
               fid = fopen(fullfile(dec_dir, 'anls_opts.txt'), 'wt');
               fprintf(fid,'%s\n',anls_opts.describeParams());
               fclose(fid);

               anls_opts.getJSON(fullfile(dec_dir, 'anls_opts.json'));
           else
               fprintf('No analysis\n');
           end
           
           if ~isempty(dec_opts) 
               if ~isa(dec_opts, 'CS_DecParams')
                   dec_opts = CS_DecParams(dec_opts);
               end
               fprintf('%s\n',dec_opts.describeParams(prefix));

               fid = fopen(fullfile(dec_dir, 'dec_opts.txt'), 'wt');
               fprintf(fid,'%s\n',dec_opts.describeParams());
               fclose(fid);

               dec_opts.getJSON(fullfile(dec_dir, 'dec_opts.json'));
           else
               fprintf('No decoding');
           end

           
           cs_vid_io = CSVideoCodecInputOutputData();
           if NewCSVideoDecoder.parallel_files
               fdef_list = files_def.getAllFiles();
               results = cell(size(fdef_list));
               dec_opts_fdef = cell(size(fdef_list));
               anls_opts_fdef = cell(size(fdef_list));
               for k=1:length(fdef_list)
                   dec_opts_fdef{k} = dec_opts;
                   anls_opts_fdef{k} = anls_opts;
               end
               parfor k=1:length(fdef_list)
                   fldef = fdef_list(k);
                   anls_def = anls_opts_fdef{k};
                   dec_def = dec_opts_fdef{k};
                   if isfield(dec_def,'init') && dec_def.init == 2
                       dec_def.ref = fldef.input;
                   end
                   dec = NewCSVideoDecoder(fldef, anls_def, dec_def);
                   dec.setPrefix(prefix);
                   result= dec.run(sprintf('%s%d]', prefix, k), fldef.enc_vid);
                   fprintf(['%s file(%d) %s decoded\n     into %s\n     '...
                       'Duration %s, cmpr duration: %f, PSNR:%f\n'],...
                       prefix, k, fldef.enc_vid, fldef.output, dec.total_dur,...
                       dec.cmpr_dur, result.psnr);
                   results{k} = result;
               end
               for k=1:length(fdef_list)
                   cs_vid_io.add(results{k});
               end
               if ~isempty(fdef_list)
                   cs_vid_io.msrmnt_input_ratio = ...
                       results{1}.msrmnt_input_ratio;
                   cs_vid_io.qntzr_ampl_stddev = ...
                       results{1}.qntzr_ampl_stddev;
                   cs_vid_io.qntzr_wdth_mltplr = ...
                       results{1}.qntzr_wdth_mltplr;
               end
            else
               indx = files_def.init_getFiles();
               k=1;
               while true
                   [fldef, indx] = files_def.getFiles(indx);
                   if isequal(fldef,[])
                       if ~ isempty(dec)
                           cs_vid_io.msrmnt_input_ratio = ...
                               dec.info.enc_opts.msrmnt_input_ratio;
                           cs_vid_io.qntzr_ampl_stddev = ...
                               dec.info.enc_opts.qntzr_ampl_stddev;
                           cs_vid_io.qntzr_wdth_mltplr = ...
                               dec.info.enc_opts.qntzr_wdth_mltplr;
                       end
                       break
                   end
                   
                   if isfield(dec_opts,'init') && dec_opts.init == 2
                       dec_opts.ref = fldef.input;
                   end
                   dec = NewCSVideoDecoder(fldef, anls_opts, dec_opts);
                   dec.setPrefix(prefix);
                   result = dec.run(sprintf('%s%d]', prefix, k), fldef.enc_vid);
                   fprintf(['%s file(%d) %s decoded\n     into %s\n     '...
                       'Duration %s\n'],...
                       prefix, k, fldef.enc_vid, fldef.output, dec.total_dur);
                   cs_vid_io.add(result);
                   k=k+1;
               end
               
               case_time = toc(case_start);
               fprintf('%s Case duration: %f\n', prefix, case_time);
           end
       end
       
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
       %     output_id - Encoded input directory name.  Tnis is inserted
       %                 into output_dir in io_def.
       %     dec_id - Decoded output directory name. Inserted into dec_dir
       %              in io_def. If it contains an asterisk the asterisk is
       %              replaced by a time-stamp.
       %     io_def   - specifies the base file names and directories.
       %     io_types - (optional) may be added to restrict the output types.
       function simul_io_data = doSimulation(anls_opts, dec_opts, output_id, dec_id,...
               io_def, io_types)
           %start the stop watch
           simulation_start=tic;
           
           simul_io_data=SimulationInputOutputData();

           if ~isempty(anls_opts) && ~isa(anls_opts,'CS_AnlsParams')
               anls_opts = CS_AnlsParams(anls_opts);
           end
           
           if ~isempty(dec_opts) && ~isa(dec_opts, 'CS_DecParams')
               dec_opts = CS_DecParams(dec_opts);
           end

           % Initialize input files
           dec_id = regexprep(dec_id, '*',datestr(now,'yyyymmdd_HHMM'),'once');
           if nargin < 6
               files_def = NewCSVideoDecoder.set_output(io_def, output_id, dec_id);
           else
               files_def = NewCSVideoDecoder.set_output(io_def, output_id, dec_id,...
                   io_types);
           end
           
           case_dirs = files_def.getCaseDirs();
           n_cases = length(case_dirs);
           
           %initialize storage to record how the simulation was run and the
           %results of the simulation
           simul_data=cell(1,length(case_dirs));
           if NewCSVideoDecoder.parallel_cases
               for j = 1:NewCSVideoDecoder.parallel_cases:n_cases
                   j_end = min(j+NewCSVideoDecoder.parallel_cases-1,n_cases);
                   parfor k=j:j_end
                       prefix = sprintf('[%d]',k);
                       case_dir = case_dirs{k};
                       simul_data{k} = ...
                           NewCSVideoDecoder.doSimulationCase(...
                           anls_opts, dec_opts, files_def,...
                           struct('prefix', [prefix ']'], 'case_dir', case_dir));
                   
                       fprintf('%s Case %d of %d done\n', prefix, k, n_cases);
                   end
                   simul_io_data.setResults(simul_data(1:j_end));
                   save([files_def.outputDir() '-simul_io_data.mat'],...
                       '-mat');
               end
           else
               for k = 1:n_cases
                   prefix = sprintf('[%d]',k);
                   case_dir = case_dirs{k};
                   simul_data{k} = ...
                       NewCSVideoDecoder.doSimulationCase(...
                       anls_opts, dec_opts, files_def,...
                       struct('prefix', [prefix ']'], 'case_dir', case_dir));
                   
                   fprintf('%s Case %d of %d done\n', prefix, k, n_cases);
                   simul_io_data.setResults(simul_data(1:k));
                   save([files_def.outputDir() '-simul_io_data.mat'],...
                       '-mat');
               end
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
       function cs_vid_io = doConvertLLC_Case(conv_params, files_def, params)
           case_start = tic;
           
           if nargin < 4
               params = struct();
           end
           
           if ~isfield(params, 'prefix')
               prefix = '';
           else
               prefix = params.prefix;
           end
           
           if isa(files_def, 'FilesDef')
               if isfield(params, 'io_params') && isfield(params.io_params,...
                       'identifier')
                   files_def.specifyDirs(params.io_params.identifier);
               end
           elseif isfield(params, 'io_params')
               files_def = FilesDef(files_def, params.io_params);
           else
               files_def = FilesDef(files_def);
           end
           if isfield(params, 'case_dir')
               files_def.setDirs(struct('case', params.case_dir));
           end
           
           fprintf('%s Starting simulation case on: %s\n%s output directory=%s \n', ...
               prefix, files_def.caseDir(),  prefix, files_def.getDecoderDir());

           cs_vid_io = CSVideoCodecInputOutputData();
           cs_vid_io.clearResults();
           if NewCSVideoDecoder.parallel_files
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
                   dec = NewCSVideoDecoder(fldef, struct());
                   dec.setPrefix(dec_prefix);
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
                   dec = NewCSVideoDecoder(fldef, struct());
                   dec.setPrefix(dec_prefix);
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
       %     io_types - (optional) may be added to restrict the output types.
       function simul_io_data = doConvertLLC(conv_params, output_id, dec_id,...
               io_def, io_types)
           %start the stop watch
           simulation_start=tic;
           
           simul_io_data=SimulationInputOutputData();

           % Initialize input files
           if nargin < 5
               files_def = NewCSVideoDecoder.set_output(io_def, output_id, dec_id);
           else
               files_def = NewCSVideoDecoder.set_output(io_def, output_id, dec_id,...
                   io_types);
           end
           
           case_dirs = files_def.getCaseDirs();
           n_cases = length(case_dirs);
           
           %initialize storage to record how the simulation was run and the
           %results of the simulation
           simul_data=cell(1,length(case_dirs));
           
           if NewCSVideoDecoder.parallel_cases
               for j = 1:NewCSVideoDecoder.parallel_cases:n_cases
                   j_end = min(j+NewCSVideoDecoder.parallel_cases-1,n_cases);
                   parfor k=j:j_end
                       prefix = sprintf('[%d]',k);
                       case_dir = case_dirs{k};
                       simul_data{k} = ...
                           NewCSVideoDecoder.doConvertLLC_Case(conv_params,...
                           files_def, case_dir, ...
                           struct('prefix', [prefix ']'], 'case_dir', case_dir));
                   
                       fprintf('%s Case %d of %d done\n', prefix, k, n_cases);
                   end
                   simul_io_data.setResults(simul_data(1:j_end));
                   save([files_def.outputDir() '-simul_io_data.mat'],...
                       '-mat');
               end
           else
               for k = 1:n_cases
                   prefix = sprintf('[%d]',k);
                   case_dir = case_dirs{k};
                   simul_data{k} = ...
                       NewCSVideoDecoder.doConvertLLC_Case(conv_params,...
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
      
      % Reads a list of code sources and returns the normalized
      % measurements (except for the no_clip ones).
      function nrm_msrs = getNormMsrs(inp_list)
          nrm_msrs_step = 1000000;
          nrm_msrs_len = 0;
          nrm_msrs = [];
          for k=1:length(inp_list);
              if iscell(inp_list)
                  inp = inp_list{k};
              else
                  inp = inp_list(k);
              end
              
              fprintf('opening %s\n', inp);
              if ischar(inp)
                  input = CodeSourceFile(inp);
              else
                  input = inp;
              end
              
              dec_info = struct();
              
              while true
                  tstart = tic;
                  [code_elmnt, ~, dec_info] = ...
                      CodeElement.readElement(dec_info, input);
                  if ischar(code_elmnt)
                      exc = MException('NewCSVideoDecoder:run',...
                          ['Error in CodeElement:readElement(): '...
                          , code_elmnt]);
                      throw(exc);
                  elseif isscalar(code_elmnt) && code_elmnt == -1
                      break;
                  elseif isa(code_elmnt, 'UniformQuantizer')
                      dec_info.quantizer = code_elmnt;
                      continue;
                  elseif isa(code_elmnt, 'VidRegion')
                      dec_info.vid_region = code_elmnt;
                      continue;
                  elseif isa(code_elmnt, 'CS_EncParams')
                      dec_info.Yblk_size = code_elmnt.blk_size;
                      dec_info.enc_opts = code_elmnt;
                      initVBlocker();
                  elseif isa(code_elmnt, 'RawVidInfo')
                      dec_info.raw_vid = code_elmnt;
                      initVBlocker();
                  elseif isa(code_elmnt, 'QuantMeasurements')
                      [msrmnts, clipped_indices] = ...
                          dec_info.quantizer.unquantize(code_elmnt);
                      msrmnts(clipped_indices) = [];
                      msrmnts(code_elmnt.n_no_clip) = [];
                      msrmnts = ...
                          (msrmnts - code_elmnt.mean_msr)/code_elmnt.stdv_msr;
                      if nrm_msrs_len + length(msrmnts) > length(nrm_msrs);
                          new_size = nrm_msrs_step * ...
                              ceil((nrm_msrs_len + length(msrmnts)) / nrm_msrs_step);
                          tmp = zeros(new_size, 1);
                          tmp(1:nrm_msrs_len)= nrm_msrs(1:nrm_msrs_len);
                          nrm_msrs = tmp;
                      end
                      nrm_msrs(nrm_msrs_len+1:nrm_msrs_len+length(msrmnts)) =...
                          msrmnts;
                      nrm_msrs_len = nrm_msrs_len + length(msrmnts);
                      b_indx = dec_info.vid_region.blk_indx(1,:);
                      fprintf('[%d %d %d] dur=%5.1f added %d len=%d size=%d\n',...
                          b_indx(1), b_indx(2), b_indx(3),...
                          toc(tstart), length(msrmnts), nrm_msrs_len, length(nrm_msrs)); 
                  end
              end
          end
          nrm_msrs = nrm_msrs(1:nrm_msrs_len);
          
          function initVBlocker()
              if ~isfield(dec_info,'blocker') && isfield(dec_info, 'raw_vid')...
                      && ~isempty(dec_info.enc_opts)
                  
                  %calculate the dimensions of the read in video
                  Ysz=[...
                      dec_info.raw_vid.height,...
                      dec_info.raw_vid.width,...
                      dec_info.enc_opts.n_frames];
                  
                  if dec_info.raw_vid.UVpresent
                      UVsz = [dec_info.raw_vid.UVheight,...
                          dec_info.raw_vid.UVwidth,...
                          dec_info.enc_opts.n_frames];
                      
                      dec_info.raw_size = [Ysz; UVsz; UVsz];
                  else
                      dec_info.UVblk_size= [0,0,0];
                      dec_info.raw_size = Ysz;
                  end
                  raw_size = dec_info.raw_size;
                  if ~dec_info.enc_opts.process_color
                      raw_size = raw_size(1,:);
                  end
                  
                  dec_info.vid_blocker = VidBlocker(dec_info.Yblk_size, struct(...
                      'vid_size', raw_size,...
                      'ovrlp', dec_info.enc_opts.blk_ovrlp,...
                      'zext', [dec_info.enc_opts.zero_ext_b; ...
                      dec_info.enc_opts.zero_ext_f],...
                      'w_blk', dec_info.enc_opts.wnd_blk_enc,...
                      'w_type', dec_info.enc_opts.wnd_type,...
                      'fps', dec_info.raw_vid.fps));
                  
              end
          end
          
      end
    end
    
    methods (Static)
       function dec_data = get_cs_msrsDecData(dec_data)
            start_time = tic;
            
            [cs_msrs, clipped_indices] = ...
                dec_data.info.quantizer.unquantize(dec_data.q_msrs);
            dec_data.cs_msrs = ...
                dec_data.info.sens_mtrx.unsortNoClip(cs_msrs);
            dec_data.dc_val = dec_data.info.sens_mtrx.getDC(cs_msrs);
            dec_data.info.sens_mtrx.setZeroedRows(clipped_indices);
            
            dec_data.proc_dur = toc(start_time);
       end
       
       function dec_data = calculateDecData(dec_data)
            start_time = tic;
            
            cs_msrs = dec_data.cs_msrs;
            
            analysis_needed = dec_data.dec_anls || dec_data.dec_mrk_blks ||...
                dec_data.dec_slct_mrk_blks || dec_data.dec_mrk_blks ||...
                dec_data.ref_mrk_blks;
                        
            if analysis_needed
                % b_indx = dec_data.info.vid_region.blk_indx;
                
                % comp background statistic
                %bg_diff = dec_data.cs_msrs-dec_data.y_bg_avg;
                %dec_data.y_bg_log_l2distance = log(dot(bg_diff, bg_diff));
                
                % compute the statistic to compare with a threshold
                dec_data.activity_statistic =abs(dec_data.y_bg_log_l2distance...
                    -dec_data.y_bg_mean_log_l2distance)...
                    /sqrt(dec_data.y_bg_var_log_l2distance);
                fprintf('activity_stat=%d\n',...
                    dec_data.activity_statistic);
                
                dec_data.alternative_statistic =abs(dec_data.y_bg_log_l2distance...
                    -log(length(dec_data.cs_msrs(2:end))*mean(dec_data.y_bg_const_var)))...
                    /sqrt(2/length(dec_data.cs_msrs(2:end)));
                fprintf('alternative_stat=%d\n',...
                    dec_data.alternative_statistic);
                
                activity_threshold = 2;
                activity_Altthreshold = 50;
                dec_data.has_activity = ...
                    (dec_data.activity_statistic > activity_threshold && dec_data.alternative_statistic > activity_Altthreshold);
                
                dec_data.blk_motion = NewCSVideoDecoder.analyzeMsrs(...
                    dec_data.info, dec_data.anls_opts, dec_data.prfx, cs_msrs);
                has_motion = ~isempty(dec_data.blk_motion) && ...
                    dec_data.blk_motion.motionFound();
            else
                has_motion = false;
            end
            
            pxmx = dec_data.info.raw_vid.getPixelMax();
            
            slct_reconstruct = dec_data.dec_slct_mrk_blks && ...
                (has_motion ||  dec_data.has_activity) && ...
                dec_data.blk_motion.maxval >= 0;
            
            if slct_reconstruct && dec_data.anls_opts.ignore_edge
                blk_cnt = dec_data.info.vid_blocker.calcBlkCnt();
                slct_reconstruct = false;
                for k=1:size(dec_data.info.vid_region.blk_indx,1)
                    bi = dec_data.info.vid_region.blk_indx(k,1:2);
                    if bi(1)>1 && bi(1)<blk_cnt(1) && ...
                            bi(2)>1 && bi(2)<blk_cnt(2)
                        slct_reconstruct = true;
                        break;
                    end
                end
            end
            
            reconstruction_needed = dec_data.dec_blks || dec_data.pre_diff_blks ||...
                dec_data.dec_mrk_blks || slct_reconstruct;
            
            % Check if there is any output file which needs reconstruction
            slct_blks = [];
            if reconstruction_needed
                % Do reconstruction
                [dec_blks, pre_diff_blks, ~] = ...
                    NewCSVideoDecoder.reconstructVideo(dec_data.info, dec_data.solver_opts,...
                    dec_data.prfx, cs_msrs, dec_data.ref_blks);
                
                % Create dec_blks if necessary
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
                
                % Create dec_mrk_blks if necessary
                if dec_data.dec_mrk_blks
                    dec_data.dec_mrk_blks = mark_blk_boundaries(dec_blks,...
                        dec_data.info.vid_blocker.ovrlp, dec_data.info.enc_opts.conv_rng,...
                        0, pxmx);
                    %dec_data.dec_mrk_blks = dec_data.info.vid_region.drawMotionMarker(...
                     %   dec_data.dec_mrk_blks, [0.5,0.5], [0, pxmx], ...
                      %  dec_data.blk_motion);
                   
                else
                    dec_data.dec_mrk_blks = [];
                end
                
                % Create the selected blocks slct_mrk_blks, if motion
                % happened
                if slct_reconstruct
                   slct_blks = dec_data.info.vid_region.drawMotionMarker(...
                       dec_blks, [0.5,0.5], [0, pxmx], ...
                       dec_data.blk_motion);
                    
                end
            else
                dec_data.dec_blks = [];
                dec_data.pre_diff_blks = [];
                dec_data.dec_mrk_blks = [];
            end
            
            % Create the selected blocks slct_mrk_blks,
            if dec_data.dec_slct_mrk_blks
                if isempty(slct_blks)
                    slct_blks = dec_data.info.vid_region.getEmpty();
                    slct_blks{1} = (pxmx/2)*ones(size(slct_blks{1}));
                    if has_motion
                      %  slct_blks = dec_data.info.vid_region.drawMotionMarker(...
                      %      slct_blks, [0.5,0.5], [0, pxmx], ...
                       %     dec_data.blk_motion);
                    end
                end
                dec_data.dec_slct_mrk_blks = slct_blks;
            else
                dec_data.dec_slct_mrk_blks = [];
            end
            
            % Create ref_mrk_blks
            if dec_data.ref_mrk_blks
                dec_data.ref_blks = mark_blk_boundaries(dec_data.ref_blks,...
                    dec_data.info.vid_blocker.ovrlp, dec_data.info.enc_opts.conv_rng,...
                    0, pxmx);
                %dec_data.ref_blks = dec_data.info.vid_region.drawMotionMarker(...
                    %dec_data.ref_blks, [0.5,0.5], [0, pxmx], ...
                   % dec_data.blk_motion);
                   
                  % Draw markers based on the statistics and threshold on
                   % the reference blocks
                if dec_data.activity_statistic > 0 && dec_data.activity_statistic < inf && dec_data.alternative_statistic > 50
                   dec_data.ref_blks = dec_data.info.vid_region.drawStatMotionMarker(...
                        dec_data.ref_blks, [0.5,0.5], [0, pxmx],...
                        dec_data.activity_statistic, 2);
                end
                
                    if dec_data.alternative_statistic > 0 && dec_data.alternative_statistic < inf && dec_data.activity_statistic > 2
                    dec_data.ref_blks = dec_data.info.vid_region.drawAltStatMotionMarker(...
                        dec_data.ref_blks, [0.5,0.5], [0, pxmx],...
                        dec_data.alternative_statistic, 50);
                    end
            else
                dec_data.ref_mrk_blks = [];
            end
            
            dec_data.proc_dur = dec_data.proc_dur + toc(start_time);
            
%             fprintf('%s [%s] decoded. R+P= %.3f + %.3f = %.3f sec.\n', dec_data.prfx,...
%                 int2str(dec_data.info.vid_region.blk_indx(1,:)), ...
%                 dec_data.read_dur, dec_data.proc_dur, ...
%                 (dec_data.read_dur + dec_data.proc_dur));
            
        end
        
        function blk_motion = analyzeMsrs(dec_info, opts, prfx, dec_cs_msrmnts)
            [nrmmcor, blk_motion] =...
                next_msrs_xcor(dec_cs_msrmnts, dec_info.sens_mtrx, ...
                dec_info.vid_region, opts);
            if ~isempty(nrmmcor) && blk_motion.motionFound()
                fprintf('%s [%s] Motion: %s\n', prfx,...
                    int2str(dec_info.vid_region.blk_indx(1,:)),blk_motion.report());
%                 for k=1:size(nrmmcor,3)
%                     fprintf('%s %d):\n%s\n',...
%                         prfx, k, mtrx_to_str(nrmmcor(:,:,k)));
%                 end
                
            end
        end

        function [blks, pre_diff_blks, sideinfo] = reconstructVideo(...
                dec_info, slv_opts, prfx, dec_cs_msrmnts, ref_blks)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % --> recover the original signal from compressive
            % sensing measurements
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if isfield(slv_opts, 'sparser')
                if strcmp(slv_opts.sparser, 'VidSparserTV3')
                    sparser = VidSparserTV3(dec_info.vid_region,...
                        slv_opts.sparser_circ);
                elseif strcmp(slv_opts.sparser, 'VidSparserD3')
                    sparser = VidSparserD3(dec_info.vid_region,...
                        slv_opts.sparser_circ);
                else
                    sparser = eval(slv_opts.sparser);
                end
            elseif any(dec_info.enc_opts.blk_pre_diff)
                sparser = BaseSparser(dec_info.vid_region.vec_len);
            else
                sparser = VidSparserTV_DCT(dec_info.vid_region, ...
                    slv_opts.sparser_circ);
            end
            
            sideinfo = struct();
            
            if isfield(slv_opts, 'use_old') && slv_opts.use_old
                blks = NewCSVideoDecoder.recoverInputFromCSMeasurements( ...
                    dec_info.sens_mtrx, sparser, dec_cs_msrmnts, slv_opts.use_old);
            else
                if slv_opts.init == -1
                    init_reg = ref_blks;
                    if isempty(init_reg)
                        slv_opts.init = 0;
                    else
                        if any(dec_info.enc_opts.blk_pre_diff)
                            init_reg = dec_info.vid_region.multiDiffExtnd(...
                                init_reg, dec_info.enc_opts.blk_pre_diff);
                        end
                        slv_opts.init = dec_info.vid_region.vectorize(init_reg);
                    end
                end
                
                q_step = dec_info.quantizer.qStep();
                pix_stdv_err = dec_info.quantizer.q_wdth_unit/sqrt(12);
                [p_mtrx, ~] = dec_info.vid_region.getExpandMtrx(...
                    slv_opts.expand_cnstrnt_level);
                if ~isempty(p_mtrx)
                    sens_mtrx = SensingMatrixCascade(...
                        {dec_info.sens_mtrx, p_mtrx});
                    sparser.setExpander(p_mtrx);
                else
                    sens_mtrx = dec_info.sens_mtrx;
                end
                if isfield(slv_opts, 'Q_msrmnts') && ...
                        slv_opts.Q_msrmnts && q_step ~= 0
                    if nargout >= 2
                        [xvec, blk_done, ...
                            sideinfo.lambda, sideinfo.beta, sideinfo.out] = ...
                            solveQuant(sens_mtrx, dec_cs_msrmnts,...
                            sparser, slv_opts, q_step, pix_stdv_err);
                    else
                        [xvec, blk_done] = ...
                            solveQuant(sens_mtrx, dec_cs_msrmnts,...
                            sparser, slv_opts, q_step, pix_stdv_err);
                    end
                else
                    if nargout >= 2
                        [xvec, blk_done, ...
                            sideinfo.lambda, sideinfo.beta, sideinfo.out] = ...
                            solveExact(sens_mtrx, dec_cs_msrmnts,...
                            sparser, slv_opts, q_step, pix_stdv_err);
                    else
                        [xvec, blk_done] = ...
                            solveExact(sens_mtrx, dec_cs_msrmnts,...
                            sparser, slv_opts, q_step, pix_stdv_err);
                    end
                end
                if ~blk_done
                    fprintf('%s ---- block decoding did not converge!\n', prfx);
                end
                if ~isempty(p_mtrx)
                    xvec = p_mtrx.multVec(xvec);
                end
                blks = dec_info.vid_region.pixelize(xvec);
            end
            
            if any(dec_info.enc_opts.blk_pre_diff)
                pre_diff_blks = blks;
                blks = dec_info.vid_region.undo_multiDiffExtnd(...
                    blks, dec_info.enc_opts.blk_pre_diff);
            else
                pre_diff_blks = [];
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

