function ext_vec =zeroExtnd(obj, vec)
  %zeroExtnd ext_vec =zeroExtnd(obj, vec) Performs both zero extned and
  %wrap extension
  
  if ~any([obj.zext(:); obj.wext(:)])
    if iscell(vec)
      ext_vec = obj.vectorize(vec);
    else
      ext_vec = vec;
    end
  else
    if ~iscell(vec)
      vec = obj.pixelize(vec);
    end

    zext_size = obj.clrBlkSize() + obj.zext(1,:) + ...
      obj.zext(2,:) + obj.wext;
    zbb = obj.zext(1,:)+1;
    zf = obj.zext(2,:) + obj.wext; 
    ext_vec =zeros([zext_size obj.n_blk obj.n_color]);
    
    we = obj.wext+1;
   
    for iclr = 1:obj.n_color
      for iblk=1:obj.n_blk
        ext_vec(zbb(1):end-zf(1),zbb(2):end-zf(2),zbb(3):end-zf(3),...
          iblk, iclr) = vec{iclr, iblk};
      end
    end
        
    % Applying wext where necessary
    for k=1:obj.wext(1)
      ext_vec(end-zf(1)+k,:,:,:,:) = ...
        ((we(1)-k)/we(1))*ext_vec(end-zf(1),:,:,:,:) +...
        (k/we(1))*ext_vec(1,:,:,:,:);
    end
    for k=1:obj.wext(2)
      ext_vec(:,end-zf(2)+k,:,:,:) = ...
        ((we(2)-k)/we(2))*ext_vec(:,end-zf(2),:,:,:) +...
        (k/we(2))*ext_vec(:,1,:,:,:);
    end
    for k=1:obj.wext(3)
      ext_vec(:,:,end-zf(3)+k,:,:) = ...
        ((we(3)-k)/we(3))*ext_vec(:,:,end-zf(3),:,:) +...
        (k/we(3))*ext_vec(:,:,1,:,:);
    end
  end
  
  ext_vec = ext_vec(:);
end

