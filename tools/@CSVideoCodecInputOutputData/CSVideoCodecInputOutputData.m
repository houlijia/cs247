classdef CSVideoCodecInputOutputData < CS_EncVidParams
    %contains data about how the video volume was processed and what were the
    % results of the video volume (entire video, unblocked) processing
    
    properties
        
        %output values of the compressed sensed video codec
        n_vid_blks;   % Total number of blocks processed
        ttl_frames;   % Total number of frames processed
        vid_size=[];     % Video size
        fps=0;          % frames rate (per second)
        msqr_err;     % mean squared error
        psnr;         % Peak Singnal to noise ratio
        ttl_inp_val;  % Total number of input values
        ttl_msrmnts; % Total number of CS measurements
        ttl_msrmnts_no_ac; % total number of measurements with no_clip spec
        ttl_msrmnts_outsd; % Number of measurements outsider quantizer range.
        % (meaning not in arithmetic coding
        dc_mean;      % Mean of DC values
        dc_var;       % Variance of DC values
        msrmnts_no_dc_mean;  % mean of all measurements except the DC measurement
        msrmnts_no_dc_var;   % variance of all measurements except the DC measurement
        ttl_entrp_bits; % Number of bits needed according to theoretical entropy model.
        ttl_msrmnts_bits; % Number of bits spent on the measurements
        ttl_entrp_clp; % Number of bits spent to indicate if clipped or not
        ttl_fxd_bits;  % Number of bits needed to represent qunatized measuremens
        % by fixed size integers
        ttl_bits;  % total bits generated by coding
        bins_max;  % maximal number of bins used by arithmetic coder
        bins_min;  % minimal number of bins used by arithmetic coder
        bins_avg;  % average number of bins used by arithmetic coder
        ttl_processing_time; % Total processing time of all blocks (seconds)
        blk_processing_time; % Average block processing time (seconds)
        
    end
    
    methods
        % Constructor
        function obj = CSVideoCodecInputOutputData(def, raw_vid)
            if nargin > 0
                args = {def};
            else
                args = {};
            end
            obj = obj@CS_EncVidParams(args{:});
            if nargin >= 2
                obj.fps = raw_vid.fps;
                obj.vid_size = [raw_vid.height, raw_vid.width, raw_vid.n_frames];
            end
            
            obj.clearResults();
        end
        
        function clearResults(obj, level)
            if nargin <2
                level = 0;
            end
            obj.n_vid_blks=0;
            obj.ttl_frames=0;
            obj.ttl_inp_val=0;
            obj.ttl_msrmnts=0;
            obj.ttl_msrmnts_no_ac=0;
            obj.ttl_msrmnts_outsd=0;
            obj.msrmnts_no_dc_mean=0;
            obj.msrmnts_no_dc_var=0;
            obj.ttl_bits=0;
            obj.ttl_msrmnts_bits = 0;
            obj.ttl_entrp_bits=0;
            obj.ttl_entrp_clp=0;
            obj.ttl_fxd_bits=0;
            obj.bins_max=0;
            obj.bins_min=0;
            obj.bins_avg=0;
            if ~level
                obj.msqr_err=0;
                obj.psnr=0;
                obj.dc_mean=0;
                obj.dc_var=0;
                obj.ttl_processing_time=0;
                obj.blk_processing_time=0;
            end
        end
        
        function calculate(cs_vid_io, cs_blk_data_list, level)
            if nargin <3
                level = 0;
            end
            
            cs_vid_io.clearResults(level);     % Clear results
            
            %intialize counters
            sum_means=0;
            sum_vars=0;
            
            %count the number of video blocks
            cs_vid_io.n_vid_blks=length(cs_blk_data_list);
            cs_vid_io.ttl_frames = cs_vid_io.n_frames;
            
            %initialize storage for dc values
            list_of_dc_values=zeros(cs_vid_io.n_vid_blks,1);
            
            %initialize storage for number of bins used by arithmetic
            %encoder
            list_n_bins=zeros(cs_vid_io.n_vid_blks,1);
            
            for iblk=1:cs_vid_io.n_vid_blks
                %extract video block data
                cs_blk_data=cs_blk_data_list{iblk};
                
                %increment the running counters
                cs_vid_io.ttl_inp_val=cs_vid_io.ttl_inp_val +...
                    cs_blk_data.n_pxl;
                cs_vid_io.ttl_msrmnts=cs_vid_io.ttl_msrmnts + ...
                    cs_blk_data.n_blk_msrmnts;
                cs_vid_io.ttl_msrmnts_no_ac=cs_vid_io.ttl_msrmnts_no_ac + ...
                    cs_blk_data.n_msrmnts_no_clip;
                cs_vid_io.ttl_msrmnts_outsd=cs_vid_io.ttl_msrmnts_outsd + ...
                    cs_blk_data.n_msrmnts_outside;
                cs_vid_io.ttl_bits=cs_vid_io.ttl_bits + ...
                    cs_blk_data.n_bits;
                cs_vid_io.ttl_msrmnts_bits = cs_vid_io.ttl_msrmnts_bits + ...
                    cs_blk_data.n_msrmnts_bits;
                cs_vid_io.ttl_entrp_bits=cs_vid_io.ttl_entrp_bits + ...
                    cs_blk_data.n_entrp_per_msrmnt * cs_blk_data.n_blk_msrmnts;
                cs_vid_io.ttl_entrp_clp = cs_vid_io.ttl_entrp_clp + ...
                    cs_blk_data.n_entrp_clipped * cs_blk_data.n_blk_msrmnts;
                cs_vid_io.ttl_fxd_bits = cs_vid_io.ttl_fxd_bits  + ...
                    ceil(log2(cs_blk_data.n_qntzr_bins) * cs_blk_data.n_blk_msrmnts);
                sum_means=sum_means + cs_blk_data.mean_msrmnts_nodc;
                sum_vars=sum_vars  + cs_blk_data.variance_msrmnts_nodc;
                list_n_bins(iblk)= cs_blk_data.n_qntzr_bins;
                
                if ~level
                    cs_vid_io.ttl_processing_time=cs_vid_io.ttl_processing_time + ...
                        cs_blk_data.seconds_to_process_block;
                    list_of_dc_values(iblk)=cs_blk_data.dc_value_of_input;
                end
                
            end
            cs_vid_io.msrmnts_no_dc_mean=sum_means / cs_vid_io.n_vid_blks;
            cs_vid_io.msrmnts_no_dc_var=sum_vars / cs_vid_io.n_vid_blks;
            
            cs_vid_io.bins_max=max(list_n_bins);
            cs_vid_io.bins_min=min(list_n_bins);
            cs_vid_io.bins_avg=mean(list_n_bins);
            
            if ~level
                cs_vid_io.dc_mean=mean(list_of_dc_values);
                cs_vid_io.dc_var=var(list_of_dc_values);
                cs_vid_io.blk_processing_time=cs_vid_io.ttl_processing_time /...
                    cs_vid_io.n_vid_blks;
            end
            
        end    % calculate
        
        function add(obj, other)
            if ~obj.ttl_frames
                obj.setParams(other)
            elseif other.ttl_frames
                ttl_frms = obj.ttl_frames + other.ttl_frames;
                if other.fps
                    if obj.fps
                        if obj.fps ~= other.fps
                            warning('FPS changed from %f to %f', obj.fps, other.fps);
                        end
                    else
                        obj.fps = other.fps;
                    end
                end
                if ~isempty(other.msqr_err)
                  obj.msqr_err = (obj.ttl_frames * obj.msqr_err + ...
                    other.ttl_frames * other.msqr_err) /ttl_frms;
                end
                if ~isempty(other.psnr)
                  obj.psnr = (obj.ttl_frames * obj.psnr + ...
                    other.ttl_frames * other.psnr) /ttl_frms;
                end

                ttl_blks = obj.n_vid_blks + other.n_vid_blks;
                obj.dc_mean = (obj.dc_mean * obj.n_vid_blks +...
                    other.dc_mean * other.n_vid_blks)/ttl_blks;
                obj.dc_var = (obj.dc_var * obj.n_vid_blks +...
                    other.dc_var * other.n_vid_blks)/ttl_blks;
                obj.msrmnts_no_dc_mean = (obj.msrmnts_no_dc_mean * obj.n_vid_blks +...
                    other.msrmnts_no_dc_mean * other.n_vid_blks)/ttl_blks;
                obj.msrmnts_no_dc_var = (obj.msrmnts_no_dc_var * obj.n_vid_blks +...
                    other.msrmnts_no_dc_var * other.n_vid_blks)/ttl_blks;
                if(other.bins_max > obj.bins_max)
                    obj.bins_max = other.bins_max;
                end
                if(other.bins_min < obj.bins_min)
                    obj.bins_min = other.bins_min;
                end
                obj.bins_avg = (obj.bins_avg * obj.n_vid_blks +...
                    other.bins_avg * other.n_vid_blks)/ttl_blks;

                obj.ttl_inp_val = obj.ttl_inp_val + other.ttl_inp_val;
                obj.ttl_msrmnts = obj.ttl_msrmnts +other.ttl_msrmnts;
                obj.ttl_msrmnts_no_ac = obj.ttl_msrmnts_no_ac +...
                    other.ttl_msrmnts_no_ac;
                obj.ttl_msrmnts_outsd = obj.ttl_msrmnts_outsd +...
                    other.ttl_msrmnts_outsd;
                obj.ttl_bits = obj.ttl_bits + other.ttl_bits;
                obj.ttl_msrmnts_bits = obj.ttl_msrmnts_bits + other.ttl_msrmnts_bits;
                obj.ttl_entrp_bits = obj.ttl_entrp_bits + other.ttl_entrp_bits;
                obj.ttl_entrp_clp = obj.ttl_entrp_clp + other.ttl_entrp_clp;
                obj.ttl_fxd_bits = obj.ttl_fxd_bits + other.ttl_fxd_bits;
                obj.ttl_processing_time = obj.ttl_processing_time +...
                    other.ttl_processing_time;

                obj.blk_processing_time = obj.ttl_processing_time / ttl_blks;
           
                obj.ttl_frames = ttl_frms;
                obj.n_vid_blks = ttl_blks;
             end
        end
        
        function []=writeToTextFile(obj, filename)
            
            fid=fopen(filename,'w');
            fprintf(fid, '%s\n\n', obj.describeParams, '');
            fprintf(fid,'number_of_video_blocks=%d\n',obj.n_vid_blks);
            fprintf(fid,'number_of_video_frame_to_read_in=%d\n',...
                obj.ttl_frames);
            
            fprintf(fid,'mean squared error=%s\n',obj.msqr_err);
            fprintf(fid,'peak signal to noise ratio=%d\n',obj.psnr);
            fprintf(fid,'No. input values=%d\n',obj.ttl_inp_val);
            fprintf(fid,'No. CS measurements=%d\n',obj.ttl_msrmnts);
            fprintf(fid,'No. CS measurements outside quantizer range =%d\n',...
                obj.ttl_msrmnts_outsd);
            fprintf(fid,...
                'No. CS measurements NOT clipped and NOT arithmetic encoded=%d\n',...
                obj.ttl_msrmnts_no_ac);
            fprintf(fid,'mean of DC values=%d\n',obj.dc_mean);
            fprintf(fid,'variance of DC values=%d\n',obj.dc_var);
            fprintf(fid,'mean of non-DC measurements=%d\n',obj.msrmnts_no_dc_mean);
            fprintf(fid,'variance of non-DC measurements=%d\n',obj.msrmnts_no_dc_var);
            fprintf(fid,'total coding bits=%d\n',obj.ttl_bits);
            fprintf(fid,'total bits for coding measurments=%d\n',obj.ttl_msrmnts_bits);
            fprintf(fid,'total bits expected by entropy of Gaussian=%d\n',...
                obj.ttl_entrp_bits);
            fprintf(fid,'total bits spent on indicating clip status=%d\n',...
                obj.ttl_entrp_clp);
            fprintf(fid,'total bits expected for fixed size measurements=%d\n',...
                obj.ttl_fxd_bits);
            fprintf(fid,'max number of bins used by AC=%d\n',obj.bins_max);
            fprintf(fid,'min number of bins used by AC=%d\n',obj.bins_min);
            fprintf(fid,'avg_number_of_bins_used_by_AC=%d\n',obj.bins_avg);
            fprintf(fid,'total processing time of all blocks (sec)=%d\n',...
                obj.ttl_processing_time);
            fprintf(fid,'average block processing time(sec)=%d\n',...
                obj.blk_processing_time);
            
            fclose(fid);
            
            
        end
        
        function prcnt = percentDiscardedMeasurements(obj)
            prcnt = 100 * (obj.ttl_msrmnts_outsd/obj.ttl_msrmnts);
        end
        
        function [coded_br, entrp_br, entrp_clp_br, fxd_br] = bitRate(obj)
            dur = obj.ttl_frames/obj.fps;
            coded_br = obj.ttl_bits /dur;
            entrp_br = obj.ttl_entrp_bits /dur;
            entrp_clp_br = obj.ttl_entrp_clp /dur;
            fxd_br = obj.ttl_fxd_bits /dur;
        end
        
        
end


end
