function  x    =   ADMM_SLOPE( y, At,para,I,W,WT,RT,R)

row = para.row;
col = para.col;
CSr = para.CSr;
beta = para.beta;
eta = para.eta;

f          =    At( y );
theta_len = length(f(:));
Re_num = round(CSr*theta_len);

im = W(f);
w = reshape(im, [row,col]);
v = zeros(row,col);
im_z = v;
x = im;

for  iter = 1 : para.iter   

    if (mod(iter, 5) == 1)
        if isfield(para,'ori_im')
            Rec_im = reshape(x,[row, col]);
            PSNR     =   psnr(Rec_im, I);                
            fprintf( 'ADMM Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
        end
    end

        x = w -v + (reshape(RT(y - R(w(:)-v(:))),[row, col]))/(beta +1);
        %beta = max(beta-0.0002,0.0001);
       %beta = beta+
       % update w
       w = (beta*(x + v))./(eta + beta+eps) + im_z*eta./(beta + eta+eps);
     % update v
       v = v + 1*(x - w);
    
    % this is the update for z, basically to get \sum_i R_i^T B z_i
       f = WT(w);
       [f_sort, ind] = sort(abs(f(:)), 'descend');   
       
       if(Re_num < theta_len)
       thero = f_sort(Re_num+1);
       else
           thero = 0;
       end
      f = sign(f).*max(abs(f)-thero,0);
      %im = reshape(W_admm(f),[row,col]); 
      im_z = reshape(W(f),[row,col]); 
   
end

