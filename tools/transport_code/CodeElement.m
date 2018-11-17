classdef CodeElement  < CompMode
  %CodeElement: An abstract class describing any element of a video code
  %
  
  properties (Access=protected)
    code=0;
    mex_handle = []; % Place for handle to a corresponding mex object
  end
  
  methods (Abstract)
    % encode - write the code of the code element into a CodeStore
    %          info is a struct of additional information.
    %          len is the number of bytes used by the code, or an error
    %          string.
    len = encode(obj, code_dst, info)
    
    % decode - read up to cnt bytes form a CodeSource, parse them and
    %          configure the object accordingly.
    %          info is a struct of additional information.
    %          len is the number of bytes read. -1
    %          is returned in case EOD was found before any reading was done;
    %          An error string indicates an error
    %          in case of error.
    
    len = decode(obj, code_src, info, cnt)
  end
  
  methods
    % Constructor - adds name to type list.
    function obj = CodeElement()
      %           obj.mex_handle = ...
      %             constructCodeElement_mex(obj, obj.getConstructSpec());
    end
    
    function delete(obj)
      if ~isempty(obj.mex_handle)
        deleteCodeElement_mex(obj.mex_handle);
      end
    end
    
    
    % This is a stub for a function whicn creates a struct with enough
    % parameters to allow constructCodeElement_mex() to create a
    % correspoding object
    function spec =getConstructSpec(obj)
      spec = struct('name', obj.getConstructName());
    end
    
    % Returns the 'name' field to be used in getConstructCode.
    function name = getConstructName(~)
      name = 'CodeElement';
    end
    
    function eql = isEqual(obj, other)
      if ~strcmp(class(obj), class(other)) || ...
          ~isequal(size(obj),size(other))
        eql = false;
        return;
      elseif all(eq(obj,other))
        eql = true;
        return
      end
      
      ignore_list = obj.ignoreInEqual();
      mc = metaclass(obj);
      props = mc.PropertyList;
      for k=1:length(props)
        if props(k).Constant || strcmp(props(k).GetAccess,'private') || ...
            any(strcmp(props(k).Name, ignore_list))
          continue;
        end
        
        prop = props(k).Name;
        
        if isa(obj.(prop),'function_handle') && isa(other.(prop),'function_handle')
          continue;
        end
        
        if ~isEqual(obj.(prop), other.(prop))
          eql = false;
          return;
        end
        
      end
      
      eql = true;
    end
    
    function len = codeLength(obj, info, add_length_of_length)
      % compCodeLength - compute the length of the code.
      % If an error occurred len is an error string.
      % If add_length_of_length is present and true, the returned value includes
      % the length of the length field that may preced the code.

      if nargin > 2 && add_length_of_length
        len = obj.codeLength(info);
        len = len + CodeStore.encodeLengthUInt(len);
      elseif ~ isa(obj.code, 'CodeStoreArray')
        obj.code = CodeStoreArray;
        len = obj.encode(obj.code, info);
      else
        len = obj.code.length();
      end
    end
    
    
    % write - encode the object and write it into a CodeDest
    % Input
    %    obj - this code element
    %    info - Information struct.
    %    code_dst - optional.  If not present, obj.code is configured
    %      as a CodeStoreArray and writing is done into it.  Note that
    %      subsequent calls to write or read will assume that obj.code is up
    %      to date, hence if the CodeElement changed obj.code should be
    %      set to 0 again.  If present, the length of the code is going
    %      to be written into code_dst followed by encoded object.
    %      If code_dst is a string it is assumed to be a file name and
    %          a CodeDestFile is opened for it.
    %      If code_dst is a cell array, it is assumed that each cell
    %        contains a CodeDest object and writing is done to each of
    %        them. The same input info in used in each writing.
    %    write_key - optional. If present and true, precede the length
    %      with a key derived from code_dst.type_list.
    % Output
    %   cnt - normally the number of bytes written (including bytes
    %         written for CodeElementTypeList updates). If an error
    %         occurred it is the error string.
    %         If obj is a cell array,
    %         cnt is expected to be the same for all.
    %         If input code_dst is a cell array, the bytes used for
    %         CodeElementTypeList updates are the maximum over all
    %         code_dst entries. an error code is
    %         returned if writing fails for any CodeDest or if the cnt
    %         is not the same for all CodeDests.
    %   out_info - updated info. If input code_dst is a cell array, an error
    %          code is returned if info is not the same for all CodeDests.
    %          Generally, info should remain unchanged after the call.
    %          info may contain object derived from class handle, hence
    %          if the info object is changed by the call, a simple
    %          assignment of info to out_info may not be sufficient to keep
    %          info unchanged.  The write function makes sure that if
    %          call causes a change in out_info, this change will
    %          impact only the out_info.
    %   type_list_cnt - number of bytes used on CodeElementTypeList
    %          updates. If code_dst is a cell array, this is an array
    %          of the same size, containing the values for each
    %          element.
    
    function [cnt, out_info, type_list_cnt] = write(obj, info, code_dst, write_key)
      out_info = info;
      
      type_list_cnt = zeros(size(code_dst));
      if nargin < 3
        if isa(obj.code, 'CodeStoreArray')
          cnt = obj.code.datalen;
        else
          obj.code = CodeStoreArray;
          cnt = obj.encode(obj.code, out_info);
        end
      elseif iscell(code_dst)
        if nargin < 4
          write_key = false;
        end
        if write_key
          for k=1:numel(code_dst)
            type_list_cnt(k) = code_dst{k}.updateTypeList(obj);
            if ischar(type_list_cnt(k))
              cnt = sprintf('Error updating CodeList for CodeDest %d: %s',...
                k, type_list_cnt(k));
              return;
            end
          end
        end
        
        cnt0 = -1;
        for k=1:length(code_dst)
          [cnt, out_info] = obj.write(info, code_dst{k}, write_key);
          if ischar(cnt)
            cnt = sprintf('Error writing to CodeDest %d: %s',...
              k, cnt);
            return;
          elseif cnt0 == -1
            cnt0 = cnt;
            out_info0 = out_info;
          elseif cnt0 ~= cnt
            cnt = sprintf(['Error writing to CodeDest %d: cnt= '...
              '%d, expected %d'], k, cnt, cnt0);
            return
          elseif ~isequal(out_info,out_info0)
            cnt = sprintf(['Error writing to CodeDest %d: '...
              'different from previous info'], k);
            return
          end
        end
        cnt = cnt + max(type_list_cnt(:));
      else
        if ischar(code_dst)
          code_dst = CodeDestFile(code_dst);
        end
        if nargin > 3 && write_key
          type_list_cnt = code_dst.updateTypeList(obj);
          if ischar(type_list_cnt); return; end
          type = code_dst.type_list.getKey(obj);
          key_cnt = code_dst.writeUInt(type);
          if ischar(key_cnt); cnt = key_cnt; return; end
          key_cnt = key_cnt + type_list_cnt;
        else
          key_cnt = 0;
        end
        
        if isa(obj.code, 'CodeStoreArray')
          len_code = obj.code.datalen;
        else
          len_code = obj.codeLength(out_info);
        end
        len_cnt = code_dst.writeUInt(len_code);
        if ischar(len_cnt); cnt = len_cnt; return; end
        
        if isa(obj.code, 'CodeStoreArray')
          err = code_dst.write(obj.code.data(1:obj.code.datalen));
          if ischar(err); cnt = err; return; end;
          cnt = obj.code.datalen;
        else
          cnt = obj.encode(code_dst, out_info);
          if ischar(cnt); return; end
        end
        cnt = cnt + len_cnt + key_cnt;
      end
    end
    
    % read - read from CodeSource (including length).
    % cnt returns the total amount of bytes read or an error code.
    % Since this function changes the object it clears obj.code.
    % max_cnt is an optional limit on the number of bytes to read.
    function cnt = read(obj, info, code_src, max_cnt)
      if nargin < 4
        max_cnt = inf;
      end
      
      if ischar(code_src)
        code_src = CodeSourceFile(code_src);
      end
      obj.code = 0;
      [len, cnt] = code_src.readUInt(max_cnt);
      if ischar(len)
        cnt = len;
        return;
      elseif len == -1;
        cnt =['EOD encountered reading length of ' class(obj) '.'];
        return;
      else
        max_cnt = max_cnt - cnt;
        if max_cnt < len
          cnt = 'EOD reached';
          return;
        end
        err = obj.decode(code_src, info, len);
        if ischar(err)
          cnt = [err ' while reading ' class(obj) ' length: ', ...
            num2str(len) '.'];
        elseif err == -1;
          cnt = 'EOD reached';
        elseif err<len
          cnt = 'decode read less than specified by length';
        else
          cnt = cnt + len;
        end
      end
    end
    
    % Some CodeElement objects may contain other CodeElement objects as
    % properties. The following functions returns the list of objects,
    % their classes and their number.
    % list of such objects
    function lst = getContained(~)
      lst = {};
    end
    
    function cls_lst = getContainedClasses(obj)
      lst = obj.getContained();
      cls_lst = cell(1,length(lst));
      for k=1:length(lst)
        cls_lst{k} = class(lst{k});
      end
      cls_lst = unique(cls_lst);
    end
    
    function cnt = getContainedLength(~)
      cnt = 0;
    end
    
    % Sets code to zero in this CodeElement and in all cotained code
    % elemtns
    function clearAllCode(obj)
      for cl = obj.getContained()
        ce = cl{1};
        ce.code = 0;
      end
      obj.code = 0;
    end
    
  end   % methods
  
  methods (Static)
    % readElement - read and return an object derived from CodeElement
    %
    % Input:
    %   code_src - A CodeSource to read from. If it is a string it is
    %         interpreted as a file name.
    %   max_cnt - (optional) maximal number of bytes to read
    % Output:
    %   ce - the returned object, if successful. -1 if EOD encountered
    %        before any reading done.  Otherwise error message
    %   cnt - number of bytes read.
    %   info - updated info
    %
    function [ce, cnt, info] = readElement(info, code_src, max_cnt)
      if nargin < 3
        max_cnt = inf;
      end
      
      if ischar(code_src)
        code_src = CodeSourceFile(code_src);
      end
      
      while true
        [type, cnt] = code_src.readUInt(max_cnt);
        if ischar(type)
          ce = ['readElement failed to read key: ' type];
          return;
        elseif type==-1
          ce = -1;
          return;
        end
        max_cnt = max_cnt - cnt;
        
        if type==0   % This is a CodeElementTypeList object
          len = code_src.typ_list.read(info, code_src, max_cnt);
          if ischar(len)
            ce = sprintf('readElement failed to read TypeList: %s', len);
            return
          else
            max_cnt = max_cnt - len;
            cnt = cnt + len;
          end
        else      % Not a TypeList
          break
        end
      end
      
      ce = code_src.typ_list.instantiate(type);
      if ischar(ce)
        return;
      end
      len = ce.read(info, code_src, max_cnt);
      if ischar(len)
        ce = sprintf('readElement failed to read object of type %s: %s', ...
          code_src.typ_list.getName(type), len);
        return
      end
      cnt = cnt+len;
    end
    
    % Write out several elemnts.  This function is more efficient than calling
    % the write method for each element separately, because the typelist is
    % updated and written out once, rather than separately for each element
    %   Input
    %     info - Information struct
    %     elmnts - an cell array of the CodeElemnt objects to write
    %     code_dst - CodeDest object to write into.
    %      If code_dst is a string it is assumed to be a file name and
    %          a CodeDestFile is opened for it.
    %      If code_dst is a cell array, it is assumed that each cell
    %        contains a CodeDest object and writing is done to each of
    %        them. The same input info in used in each writing.
    %   Output
    %   cnt - normally the number of bytes written, including the
    %         maximum number of bytes written for type_list. If an error
    %         occurred it is the error string. If obj is a cell array,
    %         cnt is expected to be the same for all.
    %         If input code_dst is a cell array, an error code is
    %         returned if writing fails for any CodeDest or if the cnt
    %         is not the same for all CodeDests.
    %   out_info - updated info. If input code_dst is a cell array, an error
    %          code is returned if info is not the same for all CodeDests.
    %          Generally, info should remain unchanged after the call.
    %          info may contain object derived from class handle, hence
    %          if the info object is changed by the call, a simple
    %          assignment of info to out_info may not be sufficient to keep
    %          info unchanged.  The write function makes sure that if
    %          call causes a change in info.type_list, this change will
    %          impact only the out_info.
    %   type_list_cnt - number of bytes used on CodeElementTypeList
    %          updates. If code_dst is a cell array, this is an array
    %          of the same size, containing the values for each
    %          element.
    
    function [cnt, out_info, type_list_cnt] = ...
        writeElements(info, elmnts, code_dst)
      
      if iscell(code_dst)
        type_list_cnt = zeros(size(code_dst));
        for k=1:numel(code_dst)
          type_list_cnt(k) = code_dst{k}.updateTypeList(elmnts);
          if ischar(type_list_cnt(k))
            cnt = sprintf('Error updating CodeList for CodeDest %d: %s',...
              k, type_list_cnt(k));
            return;
          end
        end
      else
        type_list_cnt = code_dst.updateTypeList(elmnts);
        if ischar(type_list_cnt)
          cnt = sprintf('Error updating CodeList: %s', type_list_cnt);
          return;
        end
      end
      
      out_info = info;
      cnt = 0;
      for i_elmnt=1:numel(elmnts)
        elmnt = elmnts{i_elmnt};
        [cnte, out_info] = elmnt.write(out_info, code_dst, true);
        if ischar(cnte)
          cnt = cnte;
          return
        else
          cnt = cnt + cnte;
        end
      end
    end
    
  end  % methods(Static)
  
  methods (Access = protected)
    function otr = copyElement(obj)
      otr = copyElement@matlab.mixin.Copyable(obj);
      otr.mex_handle = [];
      otr.code = 0;
    end
  end
  
  methods (Static, Access=protected)
    function ign = ignoreInEqual()
      ign = {'code', 'mex_handle', 'use_gpu', 'use_single'};
    end
  end
end

