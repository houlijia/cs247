classdef BlockingInfo < CodeElement
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % If false, this is black and white, all UV properties are ignored.
        UVpresent = false;

        % All properties beloware arrays of size [1,3], representing [
        % height,width,frame]
        Nblks = [0,0,0];       % Number of blocks in each dimension
        Yblk_size = [0,0,0];
        Yblk_ovrlp = [0,0,0];
        UVblk_size = [0,0,0];
        UVblk_ovrlp = [0,0,0];
        
        %  UVblk_size=YVblk_size ./ UVratio,
        %  UVblk_ovrlp=Yblk_ovrlp ./ UVratio.
        
        UVratio = [2,2,1];
        
    end
    
    methods
        % Set parameters.  If UV ratio is missing or any of its entris is zero
        % UVpresent is set to false.
        function set(obj, Nblk, Yblk_sz, Yblk_ovlp, ratio)
            obj.Nblks = Nblk;
            obj.Yblk_size = Yblk_sz;
            obj.Yblk_ovrlp = Yblk_ovlp;
            if nargin < 5 || ~all(ratio)
                obj.UVpresent = false;
            else
                obj.UVpresent = true;
                obj.UVblk_size = Yblk_sz ./ratio;
                obj.UVblk_ovrlp = Yblk_ovlp ./ratio;
                obj.UVratio = ratio;
            end
        end
        
        function n_blk = totalBlockNumber(obj)
            n_blk = obj.Nblks(1) * obj.Nblks(2) * obj.Nblks(3);
        end
        
        function len = encode(obj, code_dst, ~)
            len = code_dst.writeUInt(uint16(obj.UVpresent));
            if ischar(len); return; end
            
            cnt = code_dst.writeUInt(...
                [obj.Nblks obj.Yblk_size obj.Yblk_ovrlp]);
            if ischar(cnt)
                len = cnt; return
            else
                len = len + cnt; 
            end
            
            if obj.UVpresent
                cnt = code_dst.writeUInt([obj.UVratio]);
                if ischar(cnt)
                    len = cnt; return
                else
                    len = len + cnt; 
                end
            end
        end
        
        function len = decode(obj, code_src, ~, max_cnt)
            if nargin < 4
                max_cnt = inf;
            end
            
            [UVp, cnt] = code_src.readUInt(max_cnt);
            if ischar(UVp) || (isscalar(UVp) && UVp == -1)
                len = UVp; return
            else
                len = cnt;
                max_cnt = max_cnt - cnt;
            end
            
            [Yval, cnt] = code_src.readUInt(max_cnt, [1,9]);
            if ischar(Yval)
                len = Yval; return;
            elseif isscalar(Yval) && Yval == -1
                len = 'unexpected end of data'; return;
            else
                len = len + cnt;
                max_cnt = max_cnt - cnt;
                Yval = double(Yval);
            end
            
            if UVp
                [ratio, cnt] = code_src.readUInt(max_cnt, [1,3]);
                if ischar(ratio)
                    len = ratio; return;
                elseif isscalar(ratio) && ratio == -1
                    len = 'unexpected end of data'; return;
                else
                    len = len + cnt;
                    ratio = double(ratio);
                end
                obj.set(Yval(1:3), Yval(4:6), Yval(7:9), ratio);
            else
                obj.set(Yval(1:3), Yval(4:6), Yval(7:9));
            end
        end
    end
    
end

