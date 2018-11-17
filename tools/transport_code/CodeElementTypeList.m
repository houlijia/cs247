classdef CodeElementTypeList < CodeElement
    %CodeElementTypeList Contains a list of all CodeElementTypes
    
    properties
        class_list;
    end
    
    methods
        % Constructor. 
        function obj=CodeElementTypeList()
            obj.class_list = {};
        end
        
        % getKey - finds the key of a class.
        % Input:
        %   cls may be a class object of the name of a class (character
        %   string).
        % Output:
        %   Key of the class (integer) or [] if not found.
        function key = getKey(obj, cls)
            if ischar(cls)
                name = cls;
            else
                name=class(cls);
            end
            if strcmp(name, class(obj))
                key = 0;
            else
                key = find(strcmp(name, obj.class_list),1);
            end
        end
        
        % function getName - get the name of type using the numeric key
        function cls_name = getName(obj, key)
            if ~key
                cls_name = class(obj);
            elseif key>0 && key<=length(obj.class_list)
                cls_name = obj.class_list{key};
            else
                cls_name = '???';
            end
        end
              
        % Instantiate an object of the type specified by key.
        function ce = instantiate(obj, key)
            name = obj.getName(key);
            ce = eval(name);
        end
            
        % Register a new type. 
        % Input
        %   obj - this object
        %   cls - either a class name (string) or an object of a class.
        function addType(obj, cls)
            if ischar(cls)
                name = cls;
            else
                name=class(cls);
            end
            if ~strcmp(name, class(obj)) && ...
                 isempty(find(strcmp(name, obj.class_list),1))
                obj.class_list{end+1}=name;
                obj.code = 0;
            end
        end
        
        function len = encode(obj, code_dest, ~)
            len = 0;
            for k=1:length(obj.class_list)
                data = uint8(obj.class_list{k});
                cnt = code_dest.writeCode(data);
                if ischar(cnt)
                    len = cnt;
                    return;
                end
                len = len + cnt;
            end
        end
           
        function len=decode(obj, code_src, ~, cnt)
            obj.class_list = {};
            len = 0;
            
            if nargin < 4
                cnt = inf;
            end
            
            while cnt > 0
                [data, nrd] = code_src.readCode();
                if ischar(data)
                    len = data;
                    return;
                elseif data == -1;
                    len = 'EOD encountered';
                    return;
                end
                obj.class_list{end+1} = char(data);
                cnt = cnt - nrd;
                len = len + nrd;
            end
        end
    end
    
    methods (Access=protected)
        % Copy
        function other=copyElement(obj)
            other=CodeElementTypeList;
            other.class_list = obj.class_list;
        end
    end    
end

