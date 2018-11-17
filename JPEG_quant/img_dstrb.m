function img_dstrb( img_file, fig_num, out_dir)
  %img_distib draws analysis of Image
  
  nrm_exp = 2;
  
  [~,img_name,~] = fileparts(img_file);
  img = imread(img_file);
  if size(img,3) > 1
    img = rgb2gray(img);
  end

  fig = figure(fig_num);
  clf;
  fig.Name = img_name;
  fig.Units = 'pixels';
  fig.Position = [200,100,750,600];
  subplot(2,2,1, 'OuterPosition', [.01 .51 .48 .48])
  imshow(img);
  title(img_name);

  img = im2double(img) - 0.5;
 
  mt_v = SensingMatrixNrWH(size(img,1), size(img,1), 0, ...
    struct('PL_mode', SensingMatrixSqr.SLCT_MODE_NONE), 'sequency');
  mt_h = SensingMatrixNrWH(size(img,2), size(img,2), 0, ...
    struct('PL_mode', SensingMatrixSqr.SLCT_MODE_NONE), 'sequency');
  mtx = SensingMatrixKron.construct({mt_h, mt_v});
  
  y = mtx.multVec(img(:));
  y = reshape(y,size(img));
  y = abs(y) .^ nrm_exp;
  yc = cumsum(cumsum(y,1,'reverse'),2,'reverse') .^ (1/nrm_exp);
  yc = yc / yc(1);
  
  v = dct2(img);
  v = abs(v) .^ nrm_exp;
  vc = cumsum(cumsum(v,1,'reverse'),2,'reverse') .^ (1/nrm_exp);
  vc = vc / vc(1);

  sbp = subplot(2,2,3, 'OuterPosition', [.01 .01 .48 .48]);
  imshow(yc);
  colormap(sbp, 'jet');
  title('WH - High freq. L2 norm');
  
  sbp = subplot(2,2,4, 'OuterPosition', [.51 .01 .48 .48]);
  imshow(vc);
  colormap(sbp, 'jet');
  title('DCT - High freq. L2 norm');
  
  subplot(2,2,2, 'XScale','log', 'XGrid','on', 'YGrid', 'on', ...
    'XTick', [1,10,100], 'OuterPosition', [.51 .51 .48 .48]);
  hold on
  yd = cum_diag(y);
  plot((1:length(yd))', yd, '-r', 'DisplayName', 'WH');
  vd = cum_diag(v);
  plot((1:length(yd))', vd, '-b', 'DisplayName', 'DCT');
  ylim([-40,0]);
  xlim([1,round(length(vd)/2)]);
  title('Residual - triangle');
  xlabel('side of triangle');
  ylabel('residual energy (dB)');
  
  lgnd = legend('show','Location','southwest');
  lgnd.FontSize = 10;
  
  if nargin == 3
    if ~exist(out_dir, 'dir')
      [ok, errmsg, errid] = mkdir(out_dir);
      if ~ok
        error(errid, 'failed to create folder %s: %s', out_dir, errmsg);
      end
    end
    savefig(gcf, fullfile(out_dir, [img_name, '.fig']));
    saveas(gcf, fullfile(out_dir, [img_name, '.jpg']));
  end
  
  function d = cum_diag(x)
    d = zeros(length(diag(x)),1);
    dd = d;
    dd(1) = sum(x(:));
    d(1) = 1;
    for i=2:length(d);
      dd(i) = dd(i-1);
      for j=1:(i-1)
        dd(i) = dd(i) - x(j,i-j);
      end
      d(i) = 10*log(dd(i) / dd(1));
    end
  end

end

