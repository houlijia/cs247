function  [im_3d, im_2d]    =   GAP_SLOPE_3D( y, para,R,RT)

% slope  reconstruction using GAP-SLOPE
% Xin Yuan, Bell Labs, eiexyuan@gmail.com

 row = para.row;
 col = para.col;

 CSr = para.CSr;
 patch = para.patch;
 step = para.step;
  
if(strcmp(para.T_s,'dct'))
    T_dct = dctmtx(patch);
else
    level = 3;
    qmf   = MakeONFilter('Daubechies',8); 
    sig_level_row = log2(patch); 
    T_dct = get_waveletMatrix(qmf,sig_level_row,level,level);
end

I2P = @(I) image2patches3d(I, patch, patch, step, step);
P2I = @(P) patches3d2image(P, row,col,step, step);

I2Bdct = @(I) dct_block2d(I, T_dct);
Bdct2I = @(P) dct_block2d(P, T_dct');


W = @(x)  myfastidct2_givenT(x,P2I,Bdct2I);
WT = @(x) myfastdct2_givenT(x,row,col,I2P,I2Bdct);

%A = @(x) R(W(x));
AT = @(x) WT(RT(x));



f          =    AT( y );
theta_len = length(f(:));
Re_num = round(CSr*theta_len);

lamada      =   para.lambda; % 1.8, 1.2-1.7
im = W(f);
y1 = zeros(size(y));
for  iter = 1 : para.iter   

    if (mod(iter, 5) == 1)
        if isfield(para,'ori_im')
            Rec_im = reshape(im,[row, col]);
            PSNR     =   psnr(Rec_im, para.ori_im);                
            fprintf( 'GAP-SLOPE Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
        end
    end
    % for kk=1:1         
    fb        =   R( im );
    y1 = y1+ (y-fb);
    im         =   im + RT( y1-R(im) );   
    % end
    
   
    if(para.d3 && iter>round(para.ratio3d*para.iter))  % if we use the 3D transformation
        if(iter == round(para.ratio3d*para.iter)+1 || mod(iter,10)==1) % the first time, we do clustering on patches
           f_3d =  image2patches3d(reshape(im,[row,col]), patch, patch, step, step);
          [cluster_indx,cluster_num,T_dct_t] = patch_cluster(f_3d,patch,para);
           if (strcmp(para.T_t,'dct'))
                cluster.eq = true;
                cluster.cluster_num = cluster_num;
                cluster.cluster_num_ext = cluster_num;
            else
                cluster.eq = false;
                cluster.cluster_num = cluster_num;
                cluster.cluster_num_ext = 2.^ceil(log2(cluster_num));
            end
           % now we get the idx for each patch
          I2Bdct3d = @(I) dct_block3d(I, patch, T_dct, T_dct_t, cluster_indx, cluster,para.cluster);
             Bdct2I3d = @(P) idct_block3d(P, patch, T_dct, T_dct_t, cluster_indx, cluster,para.cluster);
              I2dct3d = @(I) I2Bdct3d(I2P(I));
              dct2I3d = @(P) P2I(Bdct2I3d(P));
           % now we get the idx for each patch
         end
          f          =   DCT_threshold_3d_block(reshape(im,[row,col]), CSr,I2dct3d, dct2I3d,para.cluster);
         
          im = f(:);
          im_3d = f;
    else
        % for the first several iterations we use the 2d thersholding
        f = WT(im);
        [f_sort, ind] = sort(abs(f(:)), 'descend');   
        if(Re_num < theta_len)
           thero = f_sort(Re_num+1);
        else
           thero = 0;
        end
        f = sign(f).*max(abs(f)-thero,0);
        im = W(f);
        im_2d = reshape(im, [row,col]);
    end
  
  
end

