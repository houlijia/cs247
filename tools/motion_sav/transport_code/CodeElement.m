classdef CodeElement  < matlab.mixin.Copyable
    %CodeElement: An abstract class describing any element of a video code
    %   
    
    properties
        code=0;
    end
    
    methods (Abstract)
        % encode - write the code of the code element into a CodeDest
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
        end
            
        function eql = isEqual(obj, other)
            if ~strcmp(class(obj), class(other))
%                 warning('isEqual:class','isEqual class different: %s ~= %s', ...
%                     class(obj), class(other));
                eql = false;
                return;
            end
            
            otr = other.copy();
            otr.code = obj.code;
            
            props = sort(properties(obj));
            for k=1:length(props)
                prop = props{k};
                if isobject(obj.(prop)) && ismethod(obj.(prop),'isEqual')
                    if ~ obj.(prop).isEqual(otr.(prop))
%                         warning('isEqual:isEqual','isEqual(%s.%s) failed',...
%                             class(obj), prop);
                        eql = false;
                        return;
                    end
                else
                    if ~isequal(obj.(prop), otr.(prop))
%                         warning('isEqual:same','isEqual(%s.%s) failed',...
%                             class(obj), prop);
                        eql = false;
                        return
                    end
                end
            end
            
            eql = true;
        end
        
        % compCodeLength - compute the length of the code.
        % If an error occurred len is an error string.
        function len = codeLength(obj, info)
            if ~ isa(obj.code, 'CodeDestArray')
                obj.code = CodeDestArray;
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
        %      as a CodeDestArray and writing is done into it.  Note that
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
        %      with a key derived from info.type_list.
        % Output
        %   cnt - normally the number of bytes written. If an error
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

        function [cnt, out_info] = write(obj, info, code_dst, write_key)
            out_info = info;
            if nargin < 3
                if isa(obj.code, 'CodeDestArray')
                    cnt = obj.code.datalen;
                else
                    obj.code = CodeDestArray;
                    cnt = obj.encode(obj.code, out_info);
                end
            elseif iscell(code_dst)
                if nargin < 4
                    write_key = false;
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
            else
                if ischar(code_dst)
                    code_dst = CodeDestFile(code_dst);
                end
                if nargin > 3 && write_key
                    changed = false;
                    if ~isfield(out_info, 'type_list')
                        out_info.type_list = CodeElementTypeList;
                        changed  = true;
                    end
                    for cl=[class(obj) obj.getContainedClasses()]
                        ob = cl{1};
                        type = out_info.type_list.getKey(ob);
                        if isempty(type)
                            if ~changed
                                out_info.type_list = out_info.type_list.copy();
                                changed = true;
                            end
                            out_info.type_list.addType(ob);
                        end
                    end
                    if changed
                        [cnt, out_info] =...
                            out_info.type_list.write(out_info,code_dst,true);
                        if ischar(cnt); return; end
                    else
                        cnt = 0;
                    end
                    type = out_info.type_list.getKey(obj);
                    key_cnt = code_dst.writeUInt(type);
                    if ischar(key_cnt); cnt = key_cnt; return; end
                    key_cnt = key_cnt + cnt;
                else
                    key_cnt = 0;
                end
                
                if isa(obj.code, 'CodeDestArray')
                    len_code = obj.code.datalen;
                else
                    len_code = obj.codeLength(out_info);
                end
                len_cnt = code_dst.writeUInt(len_code);
                if ischar(len_cnt); cnt = len_cnt; return; end
                
                if isa(obj.code, 'CodeDestArray')
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
        function cnt = read(obj, info, code_src)
            if ischar(code_src)
                code_src = CodeSourceFile(code_src);
            end
            obj.code = 0;
            [len, cnt] = code_src.readUInt();
            if ischar(len)
                cnt = len;
            elseif len == -1;
                cnt =['EOD encountered reading length of ' class(obj) '.'];
            else
                err = obj.decode(code_src, info, len);
                if ischar(err)
                    cnt = [err ' while reading ' class(obj) ' length: ', ...
                        num2str(len) '.'];
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
        % Output:
        %   ce - the returned object, if successful. -1 if EOD encountered
        %        before any reading done.  Otherwise
        %   cnt - number of bytes read.
        %   info - updated info
        % 
        function [ce, cnt, info] = readElement(info, code_src)
            if ischar(code_src)
                code_src = CodeSourceFile(code_src);
            end
            
            [type, cnt] = code_src.readUInt();
            if ischar(type)
                ce = [type 'readElement failed to read key'];
                return;
            elseif type==-1
                ce = -1;
                return;
            end
            
            if ~isfield(info, 'type_list')
                info.type_list = CodeElementTypeList;
            end
            
            ce = info.type_list.instantiate(type);
            if ischar(ce)
                return;
            end
            len = ce.read(info, code_src);
            if ischar(len)
                ce = sprintf('readElement failed to read object of type %s: %s', ...
                    info.type_list.getName(type), len);
                return
            end
            cnt = cnt+len;
            
            if isa(ce, 'CodeElementTypeList')
                info.type_list = ce;
                info_cnt = cnt;
                [ce, cnt, info] = CodeElement.readElement(info, code_src);
                cnt = cnt + info_cnt;
            end
        end
        
        % Write out several elemnts.  This function is more efficient than calling
        % the write method for each element separately, because the typelist is
        % updated and written out once, rather than separately for each element
        %   Input
        %     info - Information struct
        %     elmnts - an cell array of the CodeElemnt objects to write
        %     code_dst - CodeDest object to write into.
        %     If code_dst is a string it is assumed to be a file name and 
        %          a CodeDestFile is opened for it.
        %      If code_dst is a cell array, it is assumed that each cell 
        %        contains a CodeDest object and writing is done to each of
        %        them. The same input info in used in each writing.
        %   Output
        %   cnt - normally the number of bytes written. If an error
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
        
        function [cnt, out_info] = writeElements(info, elmnts, code_dst)
            out_info = info;
            changed = false;
            if ~isfield(out_info, 'type_list')
                out_info.type_list = CodeElementTypeList;
                changed = true;
            end
            if ~iscell(elmnts)
                elmnts = {elmnts};
            end
            for i_elmnt=1:length(elmnts)
                elmnt = elmnts{i_elmnt};
                for cl=[class(elmnt) elmnt.getContained()]
                    ob = cl{1};
                    type = out_info.type_list.getKey(ob);
                    if isempty(type)
                        if ~changed
                            out_info.type_list = out_info.type_list.copy();
                            changed = true;
                        end
                        out_info.type_list.addType(ob);
                    end
                end
            end
            
            if changed
                [cnt, out_info] = ...
                    out_info.type_list.write(out_info, code_dst, true);
                if ischar(cnt); return; end;
            else
                cnt = 0;
            end
            
            for i_elmnt=1:length(elmnts)
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
end

