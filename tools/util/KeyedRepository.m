classdef KeyedRepository < handle
  % KeyedRepository allows one to save objects into a repository and get
  % them back using keys.
  % The repository has a maximal number of items. Whenever this number is
  % reached, putting in new items removes the oldest items.
  
  properties (Access=private)
    len=0; % Current number of items in the repository
    oldest=0; % index of oldest element
    keys = []; % Array of keys, of size [max_cnt,n_keys]
    items =[]; % Cell array of items of size [max_cnt,1]
  end
  
  methods
    function obj = KeyedRepository(n_items, n_keys)
      % Constructor. May have 0 or 2 arguments
      %   Input:
      %     n_items - maximum number of items
      %     n_keys - maximum number of keys
      if nargin > 0
        obj.init(n_items, n_keys);
      end
    end
    
    function init(obj, n_items, n_keys)
      % Initialize
      %   Input:
      %     n_items - maximum number of items
      %     n_keys - maximum number of keys
      obj.items = cell(n_items,1);
      obj.keys = zeros(n_items, n_keys);
      obj.len = 0;
      obj.oldest = 1;
    end
    
    function item = get(obj,key) 
      % Get an item using key
      if isempty(obj.items)
        item = []; % Repository not initialized
        return
      end
      indx = obj.findItem(key);
      if isempty(indx)
        item = [];
      else
        item = obj.items{indx};
      end
    end
    
    function put(obj, item, key)
      if isempty(obj.items)
        return
      end
      indx = obj.findItem(key);
      if isempty(indx)
        % Inset item)
        obj.items{obj.oldest} = item;
        obj.keys(obj.oldest,:) = key;
        obj.oldest = 1 + mod(obj.oldest, length(obj.items));
        if obj.len < length(obj.items)
          obj.len = obj.len+1;
        end
      else
        % make item the newest
        if obj.len < length(obj.items)
          newest_indx = obj.len;
        else
          newest_indx = 1 + mod(obj.oldest-2, length(obj.items));
        end
        if newest_indx ~= indx
          obj.items([newest_indx,indx]) = obj.items([indx,newest_indx]);
          obj.keys([newest_indx,indx],:) = obj.keys([indx,newest_indx],:);
        end
      end
    end
  end
  
  methods (Access=private)
    function indx = findItem(obj, key)
      % Returns the index of the object with key 'key' or empty if no such
      % key.
      indx = find(obj.keys(1:obj.len,1) == key(1));
      if size(obj.keys,2) > 1 && ~isempty(indx)
        for k=2:(size(obj.keys,2)-1);
          indx = indx(obj.keys(indx,k) == key(k));
          if isempty(indx)
            return
          end
        end
        k = size(obj.keys,2);
        idx = find(obj.keys(indx,k) == key(k),1);
        indx = indx(idx);
      end
    end
  end
end

