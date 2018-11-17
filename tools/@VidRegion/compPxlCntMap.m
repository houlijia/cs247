function compPxlCntMap(obj)
  btm = zeros(obj.n_color,3,obj.n_blk);
  top = zeros(obj.n_color,3,obj.n_blk);
  for iblk=1:obj.n_blk
    [btm(:,:,iblk),top(:,:,iblk),~,~,~,~] = ...
      obj.blkr.blkPosition(obj.blk_indx(iblk,:));
  end
  % Maximum and minimum over blocks
  mx = max(top,[],3);
  ofst = min(btm,[],3) - 1;
  mx = mx - ofst;
  for iblk=1:obj.n_blk
    btm(:,:,iblk) = btm(:,:,iblk) - ofst;
    top(:,:,iblk) = top(:,:,iblk) - ofst;
  end
  
  pcmap = cell(obj.n_color,1);
  ncpxls = zeros(size(pcmap));
  nobcpxls = zeros(obj.n_color, obj.n_blk);
  for iclr=1:obj.n_color
    if iclr > 1 && isequal(mx(iclr,:), mx(1,:))
      pcmap{iclr} = pcmap{1};
      ncpxls(iclr) = ncpxls(1);
      nobcpxls(iclr,:) = nobcpxls(1,:);
      continue
    end
    
    if obj.n_blk == 1
      pcmap{iclr} = ones(mx(iclr,:));
      ncpxls(iclr) = numel(pcmap{iclr});
      nobcpxls(iclr,1) = ncpxls(iclr);
    else
      pcmap{iclr} = zeros(mx(iclr,:));
      for iblk = 1:obj.n_blk
        b = btm(iclr,:,iblk);
        e = top(iclr,:,iblk);
        pcmap{iclr}(b(1):e(1),b(2):e(2),b(3):e(3)) = ....
          pcmap{iclr}(b(1):e(1),b(2):e(2),b(3):e(3)) + 1;
      end
      ncpxls(iclr) = length(find(pcmap{iclr}(:)));
      
      for iblk = 1:obj.n_blk
        b = btm(iclr,:,iblk);
        e = top(iclr,:,iblk);
        pxls = pcmap{iclr}(b(1):e(1),b(2):e(2),b(3):e(3));
        pxls = pxls(:);
        pxls = pxls(pxls ~= 0);
        nobcpxls(iclr,iblk) = sum(1 ./ pxls(:));
      end
    end
  end
  
  rt = obj.blkr.vid_info.intrpltRatio();
  nobcpxls(1,:) = nobcpxls(1,:)/double(rt.Y);
  nobcpxls(2:obj.n_color,:) = nobcpxls(2:obj.n_color,:)/double(rt.UV);

  obj.pxl_cnt_map = pcmap;
  obj.ofst_pxl_cnt_map = ofst;
  obj.n_clr_pxls = ncpxls;
  obj.n_pxls = sum(ncpxls);
  obj.n_orig_blk_clr_pxls = nobcpxls;
  obj.n_orig_blk_pxls = sum(nobcpxls);
end

