classdef VidRegion < CodeElement
    % VidRegion is a description of video region.  
    % It contains a list of video block indices and information about the block
    % sizes. However, the object does not contain the pixel values or
    % equivalent information.
    %
    % When method of VidRegion deal with pixels it is assumed that they are
    % organized as an array of cells indexed by (iclr,iblk) where iclr is
    % the color component (YUV) and iblk is the block number corresponding
    % to blk_indx, that is obj.blk_indx(:,iblk) is the position of the
    % block in the video volume. Each cell contains a 3D pixel array of size
    % obj.clrBlkSize()
    % 
    % VidRegion provides the function vectorize() to convert a list of blocks
    % into a vector and pixelize() to convert a vector into a sequence of blocks.
    
    properties
        n_color=1;     % Can be 1 or 3 for color or BW.
        n_blk = 0;  % No. of blocks in the region.
        
        % array of n_blk indices of size [n_blk,3].  Each row contains the
        % [vertical, horizontal, temporal] index of a block. 
        blk_indx = []; 
        
        % zero extension of a block.  It is a 2x3 array, where
        % the first row is the backward extnesion and second is
        % the forward extension.  Columns represent (H,V,T).
        zext = [0 0 0; 0 0 0];
        
        % Wrap extension of a block
        wext = [0 0 0];
                
        blkr           % A VidBlocker object
        Cblk_len=0;    % Block length of one color including interpolation & 
                       % extension
        Cblk_size = []; % Block size (without zero extension)
        vec_len=0;     % Length of vector created from pixels
        
        ext_vec_len=0; % Length of extended vector created from pixels 
                       % (including zeroes)
        ext_Cblk_len=0;% Block length including zero extension
        
        ext_Cblk_size = []; % Block size including zero extension
    end
    
    methods
        % Constructor
        %   Can have 0,2,3 or 4 arguments. If there are no arguments a dummy object
        %   is created. 
        %     Input arguments:
        %       blk_indices - value for blk_indx
        %       vid_blocker - value for obj.blkr
        %       zero_ext - optional argument for zext 
        %               [zero_ext_b; zero_ext_f].  If it has 3 rows the
        %               the third row is wrap_ext. It can also be a struct
        %               with fields 'b','f','w' specifying zero_ext_b,
        %               zero_ext_f and wrap_ext. I
        %       wrap_ext - optional argument for wext (cannot be present if
        %               wrap_ext was specified in zero_ext).
        function obj = VidRegion(blk_indices, vid_blocker, zero_ext, wrap_ext)
          switch (nargin)
            case 0
              return
            case 2
              zero_ext = obj.zext;
              wrap_ext = obj.wext;
            case 3
              if isstruct(zero_ext)
                wrap_ext = zero_ext.w;
                zero_ext = [zero_ext.b; zero_ext.f];
              elseif size(zero_ext,1) >= 3
                wrap_ext = zero_ext(3,:);
                zero_ext = zero_ext(1:2,:);
              else
                wrap_ext = obj.wext;
              end
            case 4;  % Do nothing
            otherwise
              error('Illegal number of arguments');
          end
          obj.init(blk_indices, vid_blocker, zero_ext, wrap_ext);
        end
        
        function init(obj, blk_indices, vid_blocker, zero_ext, wrap_ext)
            obj.n_blk = size(blk_indices,1);
            obj.blk_indx = blk_indices;
            obj.blkr = vid_blocker;
            obj.n_color = size(vid_blocker.blk_size,1);
            obj.zext = zero_ext;
            obj.wext = wrap_ext;
            obj.setBlksLength();
        end
        
        function empty_vid = getEmpty(obj)
            empty_vid = cell(obj.n_color, obj. n_blk);
            for iclr = 1:obj.n_color
                for iblk = 1:obj.n_blk
                    empty_vid{iclr,iblk} = zeros(obj.Cblk_size);
                end
            end
        end
        
        % Get the pixels of a video region from a raw video cell array
        % Input:
        %   obj - this VideoRegion
        %   raw_vid - A cell array of 1 or 3 entries (Y or Y,U,V)
        % Output
        %   out - The cell array of the region.
        function out = getFromRawVid(obj, raw_vid)
            out = cell(obj.n_color, obj.n_blk);
            for i_blk = 1:obj.n_blk
                blkc = obj.blkr.getSingleBlk(raw_vid, blk_ind);
                out(1:obj.n_color, iblk) = blkc(1:obj.n_color, iblk);
                    
            end
        end
        
        % Get the pixels of a video region from a blocks cell-array
        % Input:
        %   obj - this VideoRegion
        %   blk_array - A 4-dim cell array of blocks (color, vertical,
        %   horizontal, temporal)
        %   blk_start - (optional) offset block numbers in blk_array.
        %               Default = [0,0,0].
        % Output
        %   The cell array of the region.
        function out = getFromBlkArray(obj, blk_array, blk_start)
            if nargin < 3
                blk_start = [0,0,0];
            end
            
            out = cell(obj.n_color, obj.n_blk);
%             avail_colors = min(obj.n_color, size(blk_array,1));
            for i_blk = 1:obj.n_blk
                blk_ind = obj.blk_indx(i_blk,:) - blk_start;
                v_ind = blk_ind(1);
                h_ind = blk_ind(2);
                t_ind = blk_ind(3);
                for i_clr = 1:obj.n_color %avail_colors
                    out{i_clr, iblk} = double(blk_array{i_clr, v_ind, h_ind,...
                        t_ind});
                end
            end
        end
        
        % Put the pixels of a video region into a blocks cell-array
        % Input:
        %   obj - this VideoRegion
        %   vec - the pixels to put in (a vector or a cell array
        %   (blk_no,color).
        %   blk_array - A 4-dim cell array of blocks (color, vertical,
        %   horizontal, temporal) or a VidBlocksOut object.
        %   blk_start - (optional) offset block numbers in blk_array.
        %               Default = [0,0,0].  Ignored if blk_array is a
        %               VidBlocksOut object.
        % Output
        %   blk_array - The updated block array.
        function blk_array = putIntoBlkArray(obj, vec, blk_array, blk_start)
            if nargin < 4
                blk_start = [0,0,0];
            end
            
            if ~iscell(vec)
                vec = obj.pixelize(vec);
            end
            
            for i_blk = 1:obj.n_blk
                if isa(blk_array,'VidBlocksOut')
                    blk_array.insertBlk(vec(:,i_blk), obj.blk_indx(i_blk,:));
                else
                    blk_ind = obj.blk_indx(i_blk,:) - blk_start;
                    v_ind = blk_ind(1);
                    h_ind = blk_ind(2);
                    t_ind = blk_ind(3);
                    for i_clr = 1:size(blk_array,1)
                        blk_array{i_clr, v_ind, h_ind,t_ind} =...
                            vec{i_clr,i_blk};
                    end
                end
            end
        end
       
        % put ones into a 3-array, where each entry represents a blocks,
        % for each block in this region
        % Input:
        %   obj - this VideoRegion
        %   blk_flg - the the 3D array (a vector or a cell array
        %   (blk_no,color).
        %   blk_start - (optional) starting block numbers in blk_array.
        %   Default = [1,1,1].
        % Output
        %   blk_array - The updated block array.
        function blk_flg = markDone(obj, blk_flg, blk_start)
             if nargin < 4
                blk_start = [1,1,1];
            end
            blk_start = blk_start - [1,1,1];
        
            for i_blk = 1:obj.n_blk
                blk_ind = obj.blk_indx(i_blk,:) - blk_start;
                v_ind = blk_ind(1);
                h_ind = blk_ind(2);
                t_ind = blk_ind(3);
                blk_flg(v_ind, h_ind, t_ind)= true;
            end
        end
        
        % Return a single block size
        function bsize = clrBlkSize(obj)
            bsize = obj.blkr.blk_size(1,:);
        end
          
        % convert a blocks list into a vector, without the zero
        % extension.
        % Input
        %   obj:  This object
        %   blks: A cell array of size [obj.n_color, obj.n_blk].  Each cell
        %     contains a 3-D array of pixels organized as [Vertical, Horizontal,
        %     Temporal].  blks{i,j} contains the pixels of block j, compoonent i,
        %     where i=1,2,3 corresponds to Y,U,V
        % Output
        %   vec: A vector of length obj.vec_len
        function vec = vectorize(obj, blks)
          vec = zeros(obj.vec_len,1);
          bgn_indx=1;
          for iclr = 1:obj.n_color
            for iblk=1:obj.n_blk
              subvec = blks{iclr,iblk}(:);
              end_indx=bgn_indx+length(subvec);
              vec(bgn_indx:(end_indx - 1)) = subvec;
              bgn_indx = end_indx;
            end
          end
        end
        
        % Convert a vector into a array of blocks, including removal of
        % zero extension
        % Input
        %   obj:  This object
        %   vec: A vector of length obj.vec_len
        % Output
        %   blks: A cell array of size [obj.n_color, obj.n_blk].  Each cell
        %     contains a 3-D array of pixels organized as [Vertical, Horizontal,
        %     Temporal].  blks{i,j} contains the pixels of block j, component i,
        %     where i=1,2,3 corresponds to Y,U,V
        
        function blks = pixelize(obj, vec)
            blks = cell(obj.n_color, obj.n_blk);
            bgn_indx = 1;
            bsize = obj.clrBlkSize();
            for iclr = 1:obj.n_color
                blk_len = obj.Cblk_len;
                for iblk=1:obj.n_blk
                    end_indx=bgn_indx+blk_len;
                    blks{iclr,iblk} = reshape(vec(bgn_indx:end_indx-1), bsize);
                    bgn_indx = end_indx;
                end
            end
        end
        
        % Compute total number of real pixels (before extension) in the region
        % Input:
        %   obj - this object
        function n_pxl = nPxlInRegion(obj)
            n_pxl_blk = zeros(obj.n_blk,1);
            for k=1:obj.n_blk
                n_pxl_blk(k) = obj.blkr.nPxlInBlk(obj.blk_indx(k,:));
            end
            
            n_pxl = sum(n_pxl_blk);
        end
        
        % Zero pad and wrap extenda vector pixel vector by obj.zext or obj.wext
        ext_vec =zeroExtnd(obj, vec)
        
        % Computes a sensing matrix which extends the pixel vector to
        % incorporate the zero extension. It also does pre-windowing if necssary.
        % it is a sensing matrix which is cascaded with the extension
        % matrix.
        % Input:
        %   obj - this object
        %   sens_mtrx -  (optional) sensing matrix to extend. If specified
        %   the returned matrix is the cascade of the extension matrix and the
        %   original sens_mtrix.  Otherwise it is simply the extension
        %   matrix.
        extnd_mtrx = getExtndMtrx(obj, sens_mtrx)
        
        % Compute a matrix which takes the pixel vector and expands it to
        % full block size
        % Input
        %   obj - this object
        %     level - level of expansion
        %       0 - No expansion (returned matrix is empty)
        %       1 - Only extension outside boundaries (intrp_mtrx is unit
        %           matrix
        %       2 - extension and interpolation
        % Output
        %   mtrx - the expansion matrix
        %   intrp_mtrx - interpolation only matrix
        [mtrx, intrp_mtrx] = getExpandMtrx(obj, level);
        
        % Perform Extended differences (compDiffExntd) several times along
        % several dimensions.
        % Input: 
        %   obj - this object
        %   inp - the input region pixels. If it is a vector it will be
        %         pixelized.
        %   dim_cnt - a three dimensional array specifying how many times
        %        to compute the difference along each dimension
        % Output:
        %   out - output
        function out = multiDiffExtnd(obj, inp, dim_cnt)
            if iscell(inp)
                out = inp;
            else
                out = obj.pixelize(inp);
            end
            
            for k = 1:length(dim_cnt)
                for m=1:dim_cnt(k)
                    out = obj.compDiffExtnd(out, k);
                end
            end
        end
        
        function out = edgeDtct(obj, inp, dim_cnt)
            if iscell(inp)
                einp = inp;
            else
                einp = obj.pixelize(inp);
            end
            out = einp;
            cnt = prod(2*dim_cnt+1) - 1;
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                  sz0 = size(einp{iclr,iblk});
                  sz = [1 1 1];
                  sz(1:length(sz0)) = sz0;
                  esz = sz + 2*dim_cnt;
                  vb = zeros(sz);
                  ob = einp{iclr,iblk};

                  eb = zeros(esz);
                  eb(1+dim_cnt(1):end-dim_cnt(1),...
                    1+dim_cnt(2):end-dim_cnt(2),...
                    1+dim_cnt(3):end-dim_cnt(3)) = ob;
                  
                  %extend
                  for j=1:dim_cnt(1)
                    eb(j,...
                      1+dim_cnt(2):end-dim_cnt(2),...
                      1+dim_cnt(3):end-dim_cnt(3)) = ...
                      eb(1+dim_cnt(1),...
                      1+dim_cnt(2):end-dim_cnt(2),...
                      1+dim_cnt(3):end-dim_cnt(3));
                    eb(end-j+1,...
                      1+dim_cnt(2):end-dim_cnt(2),...
                      1+dim_cnt(3):end-dim_cnt(3)) = ...
                      eb(end-dim_cnt(1),...
                      1+dim_cnt(2):end-dim_cnt(2),...
                      1+dim_cnt(3):end-dim_cnt(3));
                  end
                  for j=1:dim_cnt(2)
                    eb(:,j,1+dim_cnt(3):end-dim_cnt(3)) = ...
                      eb(:,1+dim_cnt(2),...
                      1+dim_cnt(3):end-dim_cnt(3));
                    eb(:,end-j+1,1+dim_cnt(3):end-dim_cnt(3)) = ...
                      eb(:,end-dim_cnt(2),1+dim_cnt(3):end-dim_cnt(3));
                  end
                  for j=1:dim_cnt(3)
                    eb(:,:,j) = eb(:,:,1+dim_cnt(3));
                    eb(:,:,end-j+1) = eb(:,:,end-dim_cnt(3));
                  end
                  
                  for v=-dim_cnt(1):dim_cnt(1)
                    for h=-dim_cnt(2):dim_cnt(2)
                      for t =-dim_cnt(3):dim_cnt(3)
                        vb = vb + eb(1+dim_cnt(1)+v:end-dim_cnt(1)+v,...
                          1+dim_cnt(2)+h:end-dim_cnt(2)+h,...
                          1+dim_cnt(3)+t:end-dim_cnt(3)+t);
                      end
                    end
                  end
                  
                  out{iclr,iblk} = ob - vb/cnt;
                end
            end
            if ~iscell(inp)
              out = obj.vectorize(out);
            end
              
        end
        % Undo the operatrion of multiDiffExtnd.
        % Input: 
        %   obj - this object
        %   inp - the input region pixels. If it is a vector it will be
        %         pixelized.
        %   dim_cnt - a three dimensional array specifying how many times
        %        to compute the difference along each dimension
        % Output:
        %   out - output
        function out = undo_multiDiffExtnd(obj, inp, dim_cnt)
            if iscell(inp)
                out = inp;
            else
                out = obj.pixelize(inp);
            end
            
            for k = length(dim_cnt):-1:1
                for m=1:dim_cnt(k)
                    out = obj.compCumSum(out,k);
                end
            end
        end
        
        % Compute differences along one dimension
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value of this region.
        %     dim_indx - dimension along which the differece is computed.
        %     mode - (optional). Sets the mode of the difference
        %            computation:
        %            0 - inside differnces - the output dimension is
        %                smaller by one from the input dimentsion.
        %            1 - circular differences - the output dimension
        %                is the same as the input dimesnion.
        %            2 - extended - zero is added before the entries,
        %                so the first entry is preserved.
        %  Output:
        %     dff - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays, but if mode = 0
        %           the size of the dim_indx dimension is smaller by 1.  
        %           If d_vid_reg is missing
        %           dff is a vector (vectorized by d_vid_reg internally).
        %     d_vid_reg - (optional) A VidRegion object generated from dff.
        %
        function [dff, d_vid_reg] = compDiff(obj, vec, dim_indx, mode)
            if nargin < 4
                mode = 0;
            end
            if ~iscell(vec)
                vec = obj.pixelize(vec);
            end
            
            switch(mode)
                case 0
                    dff = obj.compDiffInside(vec, dim_indx);
                    d_vid_reg = obj.getDiffVidRegion(dim_indx);
                case 1
                    dff = obj.compDiffCirc(vec, dim_indx);
                    d_vid_reg = copy(obj);
                case 2
                    dff = obj.compDiffExtnd(vec, dim_indx);
                    d_vid_reg = copy(obj);
            end
        end
        
        % Compute the size of the output of compDiff
        %   Input:
        %     obj - This object
        %     dim_indx - dimension along which the differece is computed.
        %     mode - (optional). Sets the mode of the difference
        %            computation:
        %            0 - inside differnces - the output dimension is
        %                smaller by one from the input dimentsion.
        %            1 - circular differences - the output dimension
        %                is the same as the input dimesnion.
        %            2 - extended - zero is added before the entries,
        %                so the first entry is preserved.
        %  Output:
        %
        function len = compDiffLength(obj, dim_indx, mode)
            sz = obj.clrBlkSize();
            if mode == 0
                sz(dim_indx) = sz(dim_indx)-1;
            end
            len = prod(sz) * obj.n_color * obj.n_blk;
        end
        
        function d_vid_reg = getDiffVidRegion(obj, dim_indx, mode)
            if nargin >= 3 && mode 
                d_vid_reg = obj;
                return
            end
            yb_size = obj.blkr.blk_size(1,:);
            yb_size(dim_indx) = yb_size(dim_indx) - 1;
            blk_cnt = obj.blkr.calcBlkCnt();
            blk_size = obj.blkr.blk_size;
            blk_cnt = ones(size(blk_size,1),1) * blk_cnt;
            blk_size(:,dim_indx) = blk_size(:,dim_indx)-1;
            vid_size = blk_cnt .* blk_size;
            params = obj.blkr.getParams();
            params.vid_size = vid_size;
            params.ovrlp = obj.blkr.ovrlp(1,:);
            params.linient = true;
            d_vid_reg = VidRegion(obj.blk_indx, VidBlocker(yb_size, params),...
                obj.zext, obj.wext);
        end
        
        % Compute inside differences along one dimension.
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value of this region.
        %     dim_indx - dimension along which the differece is computed.
        %  Output:
        %     dff - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays, but the size of the
        %           dim_indx dimension is smaller by 1.
        %
        function dff = compDiffInside(obj, vec, dim_indx)
            dff = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};
                    dff{iclr,iblk}= diff(blk,1,dim_indx);
                end
            end
        end
        
        % Compute circular differences along one dimension. This is the same as
        % compDiff, except that the another entry is added, of the first minus last
        % entries along the required dimension.
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value of this region.
        %     dim_indx - dimension along which the differece is computed.
        %  Output:
        %     dff - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays
        %
        function dff = compDiffCirc(obj, vec, dim_indx)
            dff = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};
                    
                    % Create a vector of corresponding to the dimensions of
                    % blk, with 1 in the dim_indx entry and zeroes in all
                    % other entires
                    shift_vec = zeros(length(size(blk)),1);
                    shift_vec(dim_indx)=-1;
                    
                    % Shift blk circularly by 1 along the dim_indx dimension
                    sblk = circshift(blk,shift_vec);
                    
                    % compute the difference
                    dff{iclr,iblk}= sblk - blk;
                end
            end
        end
        
        % Compute extended differences along one dimension. 
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value of this region.
        %     dim_indx - dimension along which the differece is computed.
        %  Output:
        %     dff - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays.
        %
        function dff = compDiffExtnd(obj, vec, dim_indx)
            dff = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};

                    % Insert zeroes at the beginning of each dim_indx
                    % "row".
                    sz = size(blk);
                    sz(dim_indx)=1;
                    blk = cat(dim_indx, zeros(sz), blk);
                    dff{iclr,iblk}= diff(blk,1,dim_indx);
                end
            end
        end
        
        % Apply the transpose of the difference along one dimension to a video
        % region of a size resulting from the transpose operation.
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value difference of this region. Note that the video has
        %           one less pixel along the dim_indx dimension
        %     dim_indx - dimension along which the differece is computed.
        %     mode - (optional). Sets the mode of the difference
        %            computation:
        %            0 - inside differnces - the output dimension is
        %                smaller by one from the input dimentsion.
        %            1 - circular differences - the output dimension
        %                is the same as the input dimesnion.
        %            2 - extended - zero is added before the entries,
        %                so the first entry is preserved.
        %  Output:
        %     out - output vector
        %
        function out = compDiffTrnsp(obj, vec, dim_indx, mode)
            if nargin < 4
                mode = 0;
            end
            switch(mode)
                case 0
                    out = obj.compDiffTrnspPxl(vec, dim_indx);
                case 1
                    out = obj.compDiffCircTrnspPxl(vec, dim_indx);
                case 2
                    out = obj.compDiffExtndTrnspPxl(vec,dim_indx);
            end
        end
        
        % Apply the transpose of the difference along one dimension to a video
        % region of a size resulting from the transpose operation.
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value difference of this region. Note that the video has
        %           one less pixel along the dim_indx dimension
        %     dim_indx - dimension along which the differece is computed.
        %  Output:
        %     out - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays, but the size of the
        %           dim_indx dimension is smaller by 1.
        %
        function out = compDiffTrnspPxl(obj, vec, dim_indx)
            if ~iscell(vec)
                d_vid_reg = obj.getDiffVidRegion(dim_indx);
                vec = d_vid_reg.pixelize(vec);
            end
            out = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};
                    
                    sz = size(blk);

                    % Create a vector of corresponding to the dimensions of
                    % blk, with -1 in the dim_indx entry and zeroes in all
                    % other entires
                    shift_vec = zeros(length(sz),1);
                    shift_vec(dim_indx)=1;
                    
                    % Insert a zero slice at the end
                    sz(dim_indx) = 1;
                    blk = cat(dim_indx, blk, zeros(sz));
                    
                    % Shift blk circularly backward along the dim_indx dimension
                    sblk = circshift(blk,shift_vec);
                    
                    % compute the difference
                    out{iclr,iblk}= sblk - blk;
                end
            end
        end
        
        % Apply the transpose of circular difference along one dimension to
        %  a video region of a size resulting from the transpose operation. This is
        %  the same as compDiffTrnspPxl except that the transform is circular.
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value difference of this region. It is organized the same
        %           way as obj
        %     dim_indx - dimension along which the differece is computed.
        %  Output:
        %     out - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays.
        %
        function out = compDiffCircTrnspPxl(obj, vec, dim_indx)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
            end
            
            out = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};
                    % Create a vector of corresponding to the dimensions of
                    % blk, with -1 in the dim_indx entry and zeroes in all
                    % other entires
                    shift_vec = zeros(length(size(blk)),1);
                    shift_vec(dim_indx)=1;
                    
                    % Shift blk circularly backward along the dim_indx dimension
                    sblk = circshift(blk,shift_vec);
                    
                    % compute the difference
                    out{iclr,iblk}= sblk - blk;
                end
            end
        end
        
        
        % Apply the transpose of the extended difference along one dimension to 
        % a video region of a size resulting from the transpose operation.
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value difference of this region. Note that the video has
        %           one less pixel along the dim_indx dimension
        %     dim_indx - dimension along which the differece is computed.
        %  Output:
        %     out - The output differences as a 2-dimensional cell array
        %           (color, block_no.) of 3 dimensional arrays, but the size of the
        %           dim_indx dimension is smaller by 1.
        %
        function out = compDiffExtndTrnspPxl(obj, vec, dim_indx)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
            end
            
            out = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};
                    
                    sz = size(blk);

                    % Insert a zero slice at the end
                    sz(dim_indx) = 1;
                    blk = cat(dim_indx, blk, zeros(sz));
                                        
                    out{iclr,iblk}= -diff(blk,1,dim_indx);
                end
            end
        end
        
        % Apply cumulative sum along one dimension
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value of this region.
        %     dim_indx - dimension along which the differece is computed.
        function out = compCumSum(obj, vec, dim_indx)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
            end
            out = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    out{iclr,iblk} = cumsum(vec{iclr,iblk}, dim_indx);
                end
            end
        end
        
        % Compute a transform along one dimension
        %   Input:
        %     obj - This object
        %     vec - Input video region.  If this is not a cell array, it is assumed
        %           to be a vectorized version of the data and it is pixelized
        %           before processing.  Otherwised it is assumed to be a
        %           pixel value of this region.
        %     dim_indx - dimension along which the differece is computed.
        %     trnsfrm - A function handle which receives a 2 dim matrix and
        %           performs the transform along each column
        %  Output:
        %     out - The output DCT.
        %
        function out = comp1dTrnsfrm(obj, vec, dim_indx, trnsfrm)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
            end
            out = cell(obj.n_color, obj.n_blk);
            for iclr = 1:obj.n_color
                for iblk=1:obj.n_blk
                    blk = vec{iclr,iblk};
                    
                    % Convert to a 2xdim matrix with dim_indx first
                    sz = ones(1,3);
                    szb = size(blk);
                    sz(1:length(szb))=szb;
                    tsz = sz;
                    indices = 1:length(sz);
                    if dim_indx ~= 1
                        indices([dim_indx,1])=[1,dim_indx];
                        tsz([dim_indx,1]) = sz([1,dim_indx]);
                        mtrx = permute(blk, indices);
                    end
                    mtrx = reshape(mtrx(:), tsz(1), prod(tsz(2:end)));
                    
                    %perform the transform
                    tblk = trnsfrm(mtrx);
                    
                    % Convert back
                    tblk = reshape(tblk(:), tsz);
                    if dim_indx ~= 1
                        tblk = permute(tblk, indices);
                    end
                    
                    out{iclr, iblk} = tblk;
                end
            end
        end
        
        % Convert pixel offsets to vector index offsets
        % Input:
        %   pxl_ofst - an array of size (m,3) where each row defines one
        %   offset - (V,H,T).
        %   use_zext - If true take zero extension into account
        % Output
        %   vec_ofst a vector of size(m,1) containing the offsets
        function vec_ofst = offsetPxlToVec(obj, pxl_ofst, use_zext)
            if use_zext
                cblk_size = obj.ext_Cblk_size;
            else
                cblk_size = obj.Cblk_size;
            end
            vec_ofst = pxl_ofst(:,2) + pxl_ofst(:,3)*cblk_size(2);
            vec_ofst = pxl_ofst(:,1) + vec_ofst*cblk_size(1);
        end
        
        % Draw motion marker
        % insert motion indicator into all blocks of the region
        % Input:
        %   vec - a cell array or vector
        %   mrkr_pos - Marker position in the block [h,v], where h,v are 
        %         relative vertical and horizontal positions in [0,1]
        %   clr - color of marker (in the Y component).
        %   cnfdnc - confidence level (pair of numbers between -1 and 1)
        %   vlcty - Velocity [h,v] in units of block size fraction per frame
        %           (i.e. motion in pixels per frame divided by block size in
        %           pixels in that dimension.
        function vec = drawMotionMarker(obj, vec, mrkr_pos, clr, blk_motion)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
                vec_is_vector = true;
            else
                vec_is_vector = false;
            end
            
           for iblk=1:size(obj.blk_indx,1)
               vec(:,iblk) = mark_blk_motion(vec(:,iblk), ...
                   mrkr_pos, clr, [blk_motion.maxval, blk_motion.midval],...
                   blk_motion.vlcty);
            end
            
            if vec_is_vector
                vec = obj.vectorize(vec);
            end
        end
        
         function vec = drawStatMotionMarker(obj, vec, mrkr_pos, clr, activity_stat, thrhld)
        %function vec = drawMotionMarker(obj, vec, mrkr_pos, clr, blk_motion)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
                vec_is_vector = true;
            else
                vec_is_vector = false;
            end
            
           % for iblk=1:size(obj.blk_indx,1)
            %    vec(:,iblk) = mark_blk_motion(vec(:,iblk), ...
             %       mrkr_pos, clr, [blk_motion.maxval, blk_motion.midval],...
              %      blk_motion.vlcty);
            %end
            
            % Mark frames based on the statistics of background
            % measurements of blocks
             for iblk=1:size(obj.blk_indx,1)
                vec(:,iblk) = Stats_based_marker(vec(:,iblk), ...
                    mrkr_pos, clr, activity_stat,...
                     thrhld);
            end
            
            if vec_is_vector
                vec = obj.vectorize(vec);
            end
         end
        
          function vec = drawAltStatMotionMarker(obj, vec, mrkr_pos, clr, activity_stat, thrhld)
        %function vec = drawMotionMarker(obj, vec, mrkr_pos, clr, blk_motion)
            if ~iscell(vec)
                vec = obj.pixelize(vec);
                vec_is_vector = true;
            else
                vec_is_vector = false;
            end
            
           % for iblk=1:size(obj.blk_indx,1)
            %    vec(:,iblk) = mark_blk_motion(vec(:,iblk), ...
             %       mrkr_pos, clr, [blk_motion.maxval, blk_motion.midval],...
              %      blk_motion.vlcty);
            %end
            
            % Mark frames based on the statistics of background
            % measurements of blocks
             for iblk=1:size(obj.blk_indx,1)
                vec(:,iblk) = AltStats_based_marker(vec(:,iblk), ...
                    mrkr_pos, clr, activity_stat,...
                     thrhld);
            end
            
            if vec_is_vector
                vec = obj.vectorize(vec);
            end
        end
        
        % Compute the vector indices which can be shifted by pxl_ofst
        % Input:
        %   obj - this object
        %   pxl_ofst - an array of size (m,3) where each row defines one
        %   offset (V,H,T).
        % Output
        %   vec_indcs - a vector of indices which can shifted by all the
        %      shiftsin pxl_ofst without going  out of bound
        vec_indcs = inRngVec(obj, pxl_ofst)
        
        % Compute 3-D cross correlation
        %   Input:
        %     obj - this object
        %     vec - the pixels of the region on which cross correlatios
        %           need to be computed.
        %     offsets - an array of size (m,3) where each row one offset to
        %           comptue correlation on.
        %   Output
        %     xcor - the cross correlations, a vector of size (m,1),
        %            divided (point wise) by vec0_nrm*vec_nrms;
        %     vec0_nrm - the norm of the reference vector (offset zero)
        %     vec_nrms - the norms of the shifted vectors
        [xcor, vec0_nrm, vec_nrms] = compXCor(obj, vec, offsets)
        
        % Compute 3-D sum-abs-diff (L1 norm of diff) of 
        %   Input:
        %     obj - this object
        %     vec - the pixels of the region on which cross correlatios
        %           need to be computed.
        %     offsets - an array of size (m,3) where each row one offset to
        %           comptue correlation on.
        %     nrm_exp - (optional, default=1) norm exponent.
        %   Output
        %     xcor - the cross correlations, a vector of size (m,1),
        %            divided (point wise) by vec0_nrm*vec_nrms;
        %     vec0_nrm - the norm of the reference vector (offset zero)
        [xcor, vec0_nrm] = compXDiff(obj, vec, offsets, nrm_exp)

        % Compute cross correlations between a pixel in a next frame and
        % pixels in a current frames, within a certain spatial distance.
        %   Input:
        %     obj - this object
        %     vec - the pixels of the region on which cross correlatios
        %     opts - (optional) a struct containing options:
        %       fxd_trgt - If true, the same set of measurements is correlated
        %           with all offsets; else the largest possible rectangle is
        %           correlated each time.  Default: true.
        %       xcor - If true, use correlation. Otherwise use difference.
        %              default: true
        %  Output
        %     xcor - the normalized cross correlations or matched score,
        %            reshaped as a 3D array, 
        %    blk_motion - BlkMotion object describing the motion foundmeasurements
        [xcor, blk_motion] = nextFrmXCor(obj, vec, opts)
        
        % Draw a rectangle in a group of frames in each block
        % Input:
        %    obj: this object
        %    vec: The pixels of the region.  Can be either in the form of
        %         pixels array (cells) or a single vector
        %    frm_no: An array of frame numbers in which the rectangle need
        %            be drawn
        %    clr: An array of the color component indices in which the
        %         rectangles need to be drawn
        %    val: The value to set. can be a scalar or of the same length
        %         as clr.
        %    pos: position of the rectangle in the frames.  This is a 2x2 array
        %          where the first row is the beginning point of (h,v)
        %          (horizontal, vertical  or 1,2) axes and the values are
        %          relative to the frame size, i.e. between 0 and 1.
        %   lwdth: (optional)line width - a 2x2 array of line widths relative
        %            to the rectangle size. first colomn is horizontal widths
        %            and 2nd is vertical widths. The defalut width is 0, meaning
        %            one pixel
        %    keep_aspect_ratio: (optional) if true the larger dimension is reduced
        %           to keep the aspect ratio.
        % Output:
        %    vec: the modified pixels, in the same format as the input
        vec = frm_draw_rect(obj,vec,frm_no,clr,val,pos,lwdth,keep_aspect_ratio)
        
        len = encode(obj, code_dst, ~)
        len = decode(obj, code_src, info, cnt)
    end
    
    methods (Access=protected)
        
        function setBlksLength(obj)
            obj.Cblk_size = obj.clrBlkSize();
            obj.Cblk_len = prod(obj.Cblk_size);
            obj.vec_len = obj.n_blk * obj.n_color * obj.Cblk_len;
            obj.ext_Cblk_size = sum([obj.Cblk_size;obj.zext;obj.wext]);
            obj.ext_Cblk_len = prod(obj.ext_Cblk_size);
            obj.ext_vec_len = obj.n_blk * obj.n_color * obj.ext_Cblk_len;
        end
        
    end
    
    
end

