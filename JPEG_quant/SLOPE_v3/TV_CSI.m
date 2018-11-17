function  im    =   TV_CSI( y, nr,para, M_func,Mt_func)
if nargin==5%function
    A=@(x) M_func(x);
    At=@(z) Mt_func(z);
else%Matrix
    A=@(x)M_func*x;
    At=@(z)M_func'*z;
end
row = para.row;
col = para.col;

im          =    At( y );
im          =    reshape(im,[row col]);
lamada      =   para.lambda;

f      =   im;
for  iter = 1 : para.iter   
     if (mod(iter, 10) == 1)
        if isfield(para,'ori_im')
            PSNR     =   psnr( f, para.ori_im(:,:,nr));                
            fprintf( 'DCT Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
        end
     end
     for ii = 1 : 3
         fb        =   A( f(:) );
         f         =   f + lamada.*reshape(At( y-fb ), [row, col]);
     end        
         f          =   TV_denoising(f,  para.TVweight,5);  
end
im     =  f;

end