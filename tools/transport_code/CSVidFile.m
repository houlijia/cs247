classdef CSVidFile < CodeElement
  %CSCodeFile name of video file
  
  properties
    name = '';
  end
  
  methods
    function obj = CSVidFile(nm)
      if nargin > 0
        obj.name = nm;
      end
    end
    
    function len = encode(obj, code_dst, ~)
      len = code_dst.writeString(obj.name);
    end
    
    function len = decode(obj, code_src, ~, cnt)
      if nargin < 4; cnt = inf; end
      
      [str, len, err_msg] = code_src.readString(cnt);
      if isnumeric(str)
        if str == -1;
          len = 'EOD found';
        else
          len = err_msg;
        end
      else
        obj.name = str;
      end
    end    
  end
  
end

