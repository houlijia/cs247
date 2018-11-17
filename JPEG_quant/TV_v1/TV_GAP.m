function  Rec_im0    =   TV_GAP( y, nr,para, M_func,Mt_func)
if nargin==5%function
    A=@(x) M_func(x);
    At=@(z) Mt_func(z);
    flag = 0;
else%Matrix
    A=@(x)M_func*x;
    At=@(z)M_func'*z;
    InvATA = inv(M_func*M_func');
    flag = 1;
end
row = para.row;
col = para.col;

if isfield(para,'T_TT_row_inv')
    flag = 2;
    T_TT_row_inv = para.T_TT_row_inv;
    T_TT_col_inv = para.T_TT_col_inv;
    ind = para.ind;
    T_row = para.T_row_sign;
    T_col = para.T_col_sign;
end

%pat = para.patch;
%addpath('./TV');


im          =    At( y );
im          =    reshape(im,[row col]);

lambda      =   para.lambda; % 1.8, 1.2-1.7
%D           =    dctmtx(pat);
y1 = zeros(size(y));
for k   =  1:1
    f      =   im;
    for  iter = 1 : para.iter   
        
        if (mod(iter, 5) == 1)
            if isfield(para,'ori_im')
                PSNR     =   psnr( f./max(f(:)), (para.ori_im(:,:,nr))./max(max(para.ori_im(:,:,nr))));                
                fprintf( 'DCT Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
            end
        end
        
        for ii = 1 : 1
            fb        =   A( f(:) );
            y1 = y1+ (y-fb);
            if(flag==1)
            f         =   f + lambda.*reshape(At(InvATA*( y1-fb) ), [row, col]);
            elseif(flag==2)
                temp1 = y1-fb;
                temp_all = zeros(row*col,1);
                temp_all(ind) = temp1;
                InvATA_E = T_TT_row_inv*reshape(temp_all, [row, col])*T_TT_col_inv';
                temp_fast = (T_row)'*InvATA_E*(T_col);
                f = f + lambda.*temp_fast;
            else
                f         =   f + lambda.*reshape(At(( y1-fb) ), [row, col]);
            end
        end        
         f          =   TV_denoising(f,  para.TVweight,5);
         
    end
    im     =  f;
end
Rec_im0   =  im;
end