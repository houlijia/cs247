classdef CodeDest < CodeStore
  % CodeDest is an abstract class representing an object into which code
  % can be written.
  % The destination can be a memory array, a file, a socket, etc.  The only
  % requirement is that it has to support sequential writing.
  % In the code, unsigned integer has a variable size representation. The
  % unsigned integer is represented as a sequence of bytes in big endian
  % (more significant bytes first) form. In each bytes the lower 7 bits
  % contain value while the MSB is a continuation bit (1 for non-final
  % byte 0 for final byte).
  % Signed integers are also supported.  The code for a signed integer is in
  % a sign-magnitude form and similar to the code for unsigned integers,
  % except that the 2nd MSB of the first byte (bit 6) is the
  % sign bit - 1 for negative and 0 for non-negative.
  % Code is usually stored in a length-value format, meaning an unsigned
  % integer in the above representation indicating the code length
  % followed by a sequence of bytes of the specified length.  Usually the
  % length is preceded by a type (key) indicating a code of what it is.
  %
  % CodeDest adds to CodeStore the element of typelist, thus allowing it to
  % generate and store keys.
  
  properties
    type_list;
  end
  
 
  methods
    
    function obj = CodeDest()
      obj.type_list = CodeElementTypeList();
    end
    
    function cnt = updateTypeList(obj, elmnts)
      % Update the type list memeber and if necessary, sends it out
      %   Input:
      %     obj: This object
      %     ce:  A CodeElement object or a cell array of CodeElement
      %          objects.
      %   Output:
      %     cnt: the number of byte use to tansmit the type_lis, if
      %          this was necessary, 0 if no transmission was necessary
      %          and an error string if an error occured.
      
      if ~iscell(elmnts)
        elmnts = {elmnts};
      end
      
      changed = false;
      for i_elmnt=1:length(elmnts)
        elmnt = elmnts{i_elmnt};
        for cl=[{elmnt} elmnt.getContained()]
          ob = cl{1};
          type = obj.type_list.getKey(ob);
          if isempty(type)
            changed = true;
            obj.type_list.addType(ob);
          end
        end
      end
      
      if changed
        [cnt, ~] = ...
          obj.type_list.write(struct(), obj, true);
      else
        cnt = 0;
      end
    end
    
    function resetTypeList(obj)
      % Clears type list
      obj.type_list = CodeElementTypeList();
    end
  end
  
  methods(Static)
    function dest = constructCodeDest(spec)
      % Create one or more CodeDest objects.
      %
      % This function creates a single CodeDest object or a cell array
      % of CodeDest objects.
      %   Input:
      %     spec - specficiation of the object(s) to create. It can be
      %     one of:
      %       - A cell array of specifications. For each cell the
      %         function generates a CodeDest object based on the
      %         specification in the cell.
      %       - A character string, in which case a CodeDestFile is
      %         created with spec as its file name
      %       - A struct, in which case a CodeDestTCP is created, based
      %         on the content of spec.
      %   Output
      %     dest - A cell array of CodeDest objects if spec is a cell
      %            array, or a single CodeDest object otherwise
      %
      
      if iscell(spec)
        dest = cell(size(spec));
        for k=1:numel(spec)
          dest{k} = CodeDest.constructCodeDest(spec{k});
        end
      elseif isa(spec, 'CodeDest')
        dest = spec;
      elseif ischar(spec)
        dest = CodeDestFile(spec);
      else
        timeout = [-1,-1];
        if isfield(spec,'sndr_is_srvr') && spec.sndr_is_srvr
          if isfield(spec,'cnct_timeout')
            timeout(1) = spec.cnct_timeout;
          else
            timeout(1) = 0;
          end
        end
        
        if isfield(spec,'linger_timeout')
          timeout(2) = spec.linger_timeout;
        end
        
        dest = CodeDestTCP(timeout, spec.addr);
      end
    end
    
    function deleteCodeDest(dst)
      if isempty(dst)
        return
      elseif iscell(dst)
        for k=1:numel(dst)
          CodeDest.deleteCodeDest(dst{k});
        end
      else
        dst.delete();
      end
    end
  end
    
end

