function  f    =   GAP_SLOPE( y, At,para,I,W,WT,Rt,R)

 row = para.row;
 col = para.col;
% pat = para.patch;
  CSr = para.CSr;

f          =    At( y );
theta_len = length(f(:));
Re_num = round(CSr*theta_len);
y1 = zeros(size(y));

im = W(f);

for  iter = 1 : para.iter   

    if (mod(iter, 5) == 1)
        %if isfield(para,'ori_im')
            Rec_im = reshape(im,[row, col]);
            PSNR     =   psnr(Rec_im, I);                
            fprintf( 'DCT Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
        %end
    end

    % gap projection
    for kk =1:1
    fb        =   R( im(:) );
    y1 = y1+ (y-fb);
    im         =   im + Rt( y1-R(im) );   
    end
   
    f = WT(im);
    
    % shrinkage 
    [f_sort, ind] = sort(abs(f(:)), 'descend');   

   if(Re_num < theta_len)
   thero = f_sort(Re_num+1);
   else
       thero = 0;
   end
  f = sign(f).*max(abs(f)-thero,0);
  im = W(f);
end

