function  Rec_im0    =   TV_ADMM( y, nr,para, M_func,Mt_func)
if nargin==5%function
    A=@(x) M_func(x);
    At=@(z) Mt_func(z);
else%Matrix
    A=@(x)M_func*x;
    At=@(z)M_func'*z;
end
row = para.row;
col = para.col;
eta = para.eta;
%pat = para.patch;
%addpath('./TV');


im          =    At( y );
im          =    reshape(im,[row col]);

lamada      =   para.lambda; % 1.8, 1.2-1.7
%D           =    dctmtx(pat);
y1 = zeros(size(y));
for k   =  1:1
    f      =   im;
    for  iter = 1 : para.iter   
        
        if (mod(iter, 5) == 1)
            if isfield(para,'ori_im')
                PSNR     =   psnr( f, para.ori_im(:,:,nr));                
                fprintf( 'DCT Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
            end
        end
        
        for ii = 1 : 1
           fb        =   A( f(:) );
            %y1 = y1+ (y-fb);
            f         =   f + reshape(At( y-fb ), [row, col])./(eta+1);
        end        
         f          =   TV_denoising(f,  para.TVweight,5);
         
    end
    im     =  f;
end
Rec_im0   =  im;
end