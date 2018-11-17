function fig_hndl = tst_img_dct2_sprs(img_file, step, fig_num)
  I=imread(img_file);
  S=im2double(I);
  if nargin < 3
    fig_hndl = figure
  else
    fig_hndl = figure(fig_num);
  end
  n_d = length(step);
  n_d1 = n_d+1;
  subplot(n_d1,3,1);
  imshow(S);
  title(img_file);
  D=dct2(S);
  D = abs(D);
  subplot(n_d1,3,2)
  imshow(D);
  title('abs 2D-DCT');
  for k=1:n_d;
    d = step(k);
    F = zeros(size(D));
    for i=1:size(D,1)
      for j=1:size(D,2)
        B = D(max(i-d,1):min(i+d,size(D,1)), max(j-d,1):min(j+d,size(D,2)));
        F(i,j) = min(B(:)) / (max(B(:)) + 1E-10);
      end
    end
    subplot(n_d1,3,k*3+1);
    imshow(F);
    title(sprintf('min/max ratio step=%d',  d));

    subplot(n_d1,3,k*3+2);
    histogram(F(:));
    title('Histogram of ratios'); 
    
    
    G = F(1:round(size(F,1)/8), 1:round(size(F,2)/8));
    subplot(n_d1,3,k*3+3);
    histogram(G(:));
    title('Hist. top left 1/8 x 1/8');  
  end
end
