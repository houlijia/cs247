classdef SAVOut < TextOut
    % SAVOut - output of records as Scene Activity Vectors (SAV)
    
    properties
        vid_blocker;  % VidBlocker object
        frames_sec;   % frames per second
        blk_cnt;      % [V,H,T]
        start_time=[];  % Reference start time
        blks_data=[];    % Contains strings corresponding to each block
        blks_data_offset;
        last_blk_indx;
        n_blks;       % Number of blocks in a frame
        start_frm;
        current_frm=0;   % current frame number
    end
    
    methods
        % Constructor. Same arguments as for superclass CSVIo
        function obj = SAVOut(vblkr)
            obj = obj@TextOut();
            obj.vid_blocker = vblkr;
            obj.frames_sec = vblkr.fps;
            obj.blk_cnt =  obj.vid_blocker.calcBlkCnt();
            obj.n_blks = obj.blk_cnt(1)*obj.blk_cnt(2);
            n_sec = ceil(obj.frames_sec/...
                (obj.vid_blocker.blk_size(1,3)-obj.vid_blocker.ovrlp(1,3)));
            obj.blks_data = cell(obj.n_blks, n_sec);
            hdr = obj.getHeader();
            cnt = obj.write(hdr);
            if ischar(cnt)
                error('Failed to write header of SAVout (%s):\n%s',...
                    cnt, hdr);
            end
        end
        
        function delete(obj)
            try
                if obj.current_frm
                    obj.writeRecord();
                end
            catch err
               % dummy action to avoid warning of unused variable
               if isfield(err,'no such field')
                   rethrow(err);
               end
            end
        end
        
        function header=getHeader(obj)
            header = sprintf(...
                ['fps=%d ;frameWidth=%d; frameHeight=%d; total level=2 ;Level 0: 1X1 ;Level 1: %dX%d ;'...
                'Types: Scene, Certainty, Direction, Velocity\n'...
                'FrameNO Date Time FrameCount S_L0B0 C_L0B0 D_L0B0 V_L0B0'], obj.frames_sec,...
                obj.vid_blocker.vid_size(1,2), obj.vid_blocker.vid_size(1,1),...
                obj.blk_cnt(2), obj.blk_cnt(1));
            
            blks = cell(1,obj.n_blks);
            for k=1:length(blks)
                ks=int2str(k-1);
                blks{k}=regexprep('S_L1Bx C_L1Bx D_L1Bx V_L1Bx ','x',ks);
            end
            
            header = sprintf('%s %s',header, horzcat(blks{:}));
        end
        
        % Input:
        %   obj - this object
        %   rec - a BlkMotion object or struct with fields
        %       indx - block index
        %       speed - motion speed (pxl/frame)
        %       direction
        %       certainty
        function setBlkRecord(obj, rec)
            rec_indx = rec.getIndex();
            [blk_bgn,blk_end,~,~,~,~] = obj.vid_blocker.blkPosition(rec_indx);
            if obj.current_frm == 0
                obj.start_frm = blk_bgn(1,3);
                obj.blks_data_offset = rec_indx(3)-1;
                obj.start_time = datevec(now);
            elseif ceil(obj.start_time(6)+obj.current_frm /obj.frames_sec) ~= ...
                    ceil(obj.start_time(6)+blk_end(1,3)/obj.frames_sec)
                obj.writeRecord();
                obj.start_frm = blk_bgn(1,3);
                obj.blks_data_offset = rec_indx(3)-1;
            end
            obj.current_frm = blk_end(1,3);
            obj.last_blk_indx = rec_indx;
            
            t = rec_indx(3) - obj.blks_data_offset;
            k = rec_indx(2) + (rec_indx(1)-1)*obj.blk_cnt(2);
            if ~isempty(obj.blks_data{k,t})
                error('block [%s] aleady set', int2str(rec_indx));
            end
            obj.blks_data{k,t} = rec;
        end   
    end
    
    methods (Access=protected)
        function writeRecord(obj)            
            n_tblks = obj.last_blk_indx(3) - obj.blks_data_offset;
            out_frms = cell(obj.n_blks,1);
            glbl = update_avg();
            for k=1:obj.n_blks
                lcl = update_avg();
                for t=1:n_tblks
                    motn = obj.blks_data{k,t};
                    if isempty(motn)
                        error('need to write while block [%d,%d] is empty',...
                            k, t);
                    end
                    glbl = update_avg(glbl, motn);
                    lcl = update_avg(lcl, motn);
                    obj.blks_data{k,t} = [];  % Mark as empty
                end
                out_frms{k} = comp_avg_str(lcl);
            end
            tm = obj.start_time;
            tm(6) = ceil(tm(6) + obj.current_frm/obj.frames_sec);
            line = sprintf('%d %s %d %s%s', obj.current_frm, ...
                datestr(tm,'mm/dd/yy HH:MM:SS'),...
                (obj.current_frm - obj.start_frm + 1), ...
                comp_avg_str(glbl), horzcat(out_frms{:}));
            cnt = obj.write(line);
            if ischar(cnt)
                error('Writing failed (%s)',cnt);
            end
            
            function avg = update_avg(avg, motn)
                if nargin == 0
                    avg = struct(...
                        'cnt',0,...
                        'v_sum', [0,0],...
                        'v_ssqr', [0,0],...
                        'crtnty', 0,...
                        'actvty', 0);
                    return
                end
                
                if motn.motionFound()
                    vl = motn.vlcty;
                    avg.v_sum = avg.v_sum + vl;
                    avg.v_ssqr = avg.v_ssqr + (vl .* vl);
                end
                avg.crtnty = avg.crtnty + motn.maxval;
                avg.actvty = avg.actvty + (1-motn.midval);
                avg.cnt = avg.cnt + 1;
            end
            
            function str = comp_avg_str(avg)
                vlcty = avg.v_sum / avg.cnt;
                crtnty = avg.crtnty / avg.cnt;
                if avg.cnt > 1 && norm(avg.v_ssqr) > 0
                    v_stdv = avg.v_ssqr - avg.cnt * (vlcty .* vlcty);
                    v_stdv = sqrt(sum(v_stdv / (avg.cnt-1)));
                    v_rms = sqrt(sum(avg.v_ssqr / (avg.cnt-1)));
                    crtnty = crtnty * (1- v_stdv / v_rms);
                end
                crtnty = max(0, min(100, round(crtnty*100)));
                actvty = avg.actvty / avg.cnt;
                actvty = max(0, min(100, round(actvty*100)));
                vlcty = complex(vlcty(2), -vlcty(1));
                speed = round(100*abs(vlcty));
                if vlcty ~= 0
                    d = angle(vlcty);
                    if d<0
                        d = d + 2*pi;
                    end
                    direction = 1+min(7, round(4*d/pi));
                else
                    direction = 0;
                end
                str = sprintf('%d %d %d %d ', actvty, crtnty, direction, speed);
            end
        end
    end
 end
