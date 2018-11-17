function [rec_im, PSNR, SSIM]   =  NLR_CS_Reconstruction_had( par, A, At  )
time0            =    clock;
y                =    par.y;
rec_im0          =    DCT_CIR( y, par, A, At );

rec_im           =    rec_im0;
AtY              =    At(y);
beta             =    par.beta;  %0.01;
M                =    zeros( size(rec_im) );
%M(par.picks)     =    1;
%M(1,1)           =    1;
%DtY              =    zeros( size(rec_im) );
%DtY(1,1)         =    y(1);
%K                =    length(y);
%DtY(par.picks)   =    y(2:(K+1)/2) + i*y((K+3)/2:K);

[h, w]           =    size( rec_im );
cnt              =    0;
iters            =    15;
for  k    =   1 : par.K
    
    blk_arr      =     Block_matching( rec_im, par);
    f            =     rec_im;
    U_arr        =     zeros(par.win^4, size(blk_arr,2), 'like', y);
    if (k<=par.K0)  flag=0;  else flag=1;  end        
    
    for it  =  1 : iters
    
        cnt      =   cnt  +  1;    
        if (mod(cnt, 40) == 0)
            if isfield(par,'ori_im')
               % PSNR     =   csnr( f, par.ori_im, 0, 0 );
                PSNR     =   psnr( rec_im./max(rec_im(:)), par.ori_im./max(par.ori_im(:)));
                fprintf( 'NLR CS Reconstruction, Iter %d : PSNR = %f \n', cnt, PSNR );
            end
        end
        
        [rim, wei, U_arr]      =   Low_rank_appro(f, par, blk_arr, U_arr, it, flag );   
        rim     =    (rim+beta*f)./(wei+beta);
        
        %if par.s_model == 1  % For the general sensing matrixes 
            b               =   AtY + beta * rim(:);
            [X, flag0]       =   pcg( @(x) Afun(x, At, A, beta, wei(:)), b, 0.5E-6, 400, [], [], f(:));          
            f               =   reshape(X, h, w);  
        
    end
    rec_im    =   f;
    % beta       =   beta * 1.05;
end

if isfield(par,'ori_im')
    %PSNR     =   csnr( rec_im, par.ori_im, 0, 0 );
      PSNR     =   psnr( rec_im./max(rec_im(:)), par.ori_im./max(par.ori_im(:)));
    SSIM      =  cal_ssim( rec_im, par.ori_im, 0, 0 );
end
% disp(sprintf('Total elapsed time = %f min\n', (etime(clock,time0)/60) ));
return;


%======================================================
function  y  =  Afun(x, At, A, eta, Wei)
y      =   At( A(x) ) + eta*x;  % eta * (Wei.*x);
return;



function   [dim, wei, U_arr]  =  Low_rank_appro(nim, par, blk_arr, U_arr, it, flag)
b            =   par.win;
[h, w, ch]    =   size(nim);
N            =   h-b+1;
M            =   w-b+1;
if isa(U_arr,'gpuArray')
  r = gpuArray.colon(1,N);
  c = gpuArray.colon(1,M);
else
  r            =   [1:N];
  c            =   [1:M]; 
end

X            =   Im2Patch( nim, par );
Ys           =   zeros( size(X), 'like', X );        
W            =   zeros( size(X), 'like', X  );
L            =   size(blk_arr,2);
T            =   4; 
for  i  =  1 : L
    B          =   X(:, blk_arr(:, i));
    if it==1 || mod(it, T)==0
        [tmp_y, tmp_w, U_arr(:,i)]   =   Weighted_SVT( cast(B, 'double'), par.c1, par.nSig^2, flag, par.c0 );
    else
        [tmp_y, tmp_w]   =   Weighted_SVT_fast( cast(B, 'double'), par.c1, par.nSig^2, U_arr(:,i), flag, par.c0 );
    end
    Ys(:, blk_arr(:,i))   =   Ys(:, blk_arr(:,i)) + tmp_y;
    W(:, blk_arr(:,i))    =   W(:, blk_arr(:,i)) + tmp_w;
end

dim     =  zeros(h,w, 'like', X  );
wei     =  zeros(h,w, 'like', X  );
k       =   0;
for i  = 1:b
    for j  = 1:b
        k    =  k+1;
        dim(r-1+i,c-1+j)  =  dim(r-1+i,c-1+j) + reshape( Ys(k,:)', [N M]);
        wei(r-1+i,c-1+j)  =  wei(r-1+i,c-1+j) + reshape( W(k,:)', [N M]);
    end
end
return;



function  [X, W, U]   =   Weighted_SVT( Y, c1, nsig2, flag, c0 )
c1                =   c1*sqrt(2);
[U0,Sigma0,V0]    =   svd(full(Y),'econ');
Sigma0            =   diag(Sigma0);
if flag==1
    S                 =   max( Sigma0.^2/size(Y, 2), 0 );
    thr               =   c1*nsig2./ ( sqrt(S) + eps );
    S                 =   soft(Sigma0, thr);
else  % use nuclear norm
    S                 =   soft(Sigma0, c0*nsig2);
end
r                 =   sum( S>0 );
U                 =   U0(:,1:r);
V                 =   V0(:,1:r);
X                 =   U*diag(S(1:r))*V';

% Weighted the reconstructed patches using the weights computed using the
% matrix ranks slightly improve the final results (less than 0.2 dB)
if r==size(Y,1)
    wei           =   1/size(Y,1);   % 1;
else
    wei           =   (size(Y,1)-r)/size(Y,1);  % 1;
end
W                 =   wei*ones( size(X), 'like', wei );
X                 =   X*wei;
U                 =   U0(:);
return;


%--------------------------------------------------------------------------
%- This function uses the PCA matrixes obtained in the previous iterations
%- to save computational complexity
%--------------------------------------------------------------------------
function  [X, W]   =   Weighted_SVT_fast( Y, c1, nsig2, U0, flag, c0 )
c1                =   c1*sqrt(2);
n                 =   sqrt(length(U0));
U0                =   reshape(U0, n, n);
A                 =   U0'*Y;
Sigma0            =   sqrt( sum(A.^2, 2) );
V0                =   (diag(1./Sigma0)*A)';

if flag==1
    S                 =   max( Sigma0.^2/size(Y, 2) - 0*nsig2, 0 );
    thr               =   c1*nsig2./ ( sqrt(S) + eps );
    S                 =   soft(Sigma0, thr);
else  
    S                 =   soft(Sigma0, c0*nsig2);
end
r                 =     sum( S>0 );
P                 =     find(S);
X                 =     U0(:,P)*diag(S(P))*V0(:,P)';
if r==size(Y,1)
    wei           =     1/size(Y,1);  % 1;
else
    wei           =     (size(Y,1)-r)/size(Y,1);  %  1;
end
W                 =     wei*ones( size(X), 'like', wei );
X                 =     X*wei;
return;



%====================================================================
% Compressive Image Recovery Using DCT 
%--------------------------------------------------------------------
function  Rec_im0    =   DCT_CIR( y, par, A, At )
%ori_im      =    par.ori_im;
%[h w]       =    size(ori_im);
im          =    At( y );
h = par.h;
w =par.w;
im          =    reshape(im,[h w]);

lamada      =    1.5;  % 1.8, 1.2-1.7
b           =    par.win*par.win;
D           =    dctmtx(b);

for k   =  1:1
    f      =   im;
    for  iter = 1 : 300   
        
        if (mod(iter, 50) == 0)
            if isfield(par,'ori_im')
               % PSNR     =   csnr( f, par.ori_im, 0, 0 );   
                PSNR     =   psnr( f./max(f(:)), par.ori_im./max(par.ori_im(:)));   
                fprintf( 'DCT Compressive Image Recovery, Iter %d : PSNR = %f\n', iter, PSNR );
            end
        end
        
        for ii = 1 : 3
            fb        =   A( f(:) );
            f         =   f + lamada.*reshape(At( y-fb ), h, w);
        end        
        f          =   DCT_thresholding( f, par, D );
    end
    im     =  f;
end
Rec_im0   =  im;
return;


