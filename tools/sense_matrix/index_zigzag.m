function Indx = index_zigzag(m,k)
  % pseudo random row selection by Xin Yuan
  % m - order of square matrix. Must be a power of 4
  % k - number of required measurements
  
  m = double(m);
  k = double(k);
  ord = log2(m);
  q = round(sqrt(m));
  indx = zeros(1,m);
  
  indx(1) = 1;
  n = 1;
  pre_n = 1;
  
  % bit reversal of 0:(m-1) plus 1.
  for i = ord-1 : -1 : 0
    for j = 1 : 2^(ord-i-1)
      n = n + 1;
      indx(n) = indx(2*pre_n-n+1)+2^i;
    end
    pre_n = n;
  end
  
  % divide indx into q groups of size q and reverse order of every other group.
  indx = reshape(indx,q,q);
  Indx = zeros(q);
  for i = 1 : q
    for j = 1 : q
      switch mod(i,2)
        case 0
          Indx(q*(i-1)+q+1-j) = indx(q*(i-1)+j); %% reversing the even-numbered row of 'indx'
        case 1
          Indx(q*(i-1)+j) = indx(q*(i-1)+j);
      end
    end
  end
  
  %% changing it into zig-zag way
  
  if k <= q*(q+1)/2 %% k <= m/2
    ini1 = 1;
    fin1 = ceil((-1+sqrt(1+8*k))/2);
  else              %% k > m/2
    ini1 = 1;
    fin1 = ceil(2*q-1/2-sqrt(2*q^2+1/4-2*k));
  end
  
  n = 0;
  a = zeros(1,k);
  b = zeros(1,k);
  for i = ini1 : fin1
    if i <= q
      ini2 = 1;
      fin2 = i;
    else
      ini2 = i+1-q;
      fin2 = q;
    end
    for j = ini2 : fin2
      n = n + 1;
      switch mod(i,2)
        case 0
          a(n) = i+1-j;
          b(n) = j;
        case 1
          a(n) = j;
          b(n) = i+1-j;
      end
      if n == k
        break
      end
    end
  end
  
  Indx = Indx(:);
  
  Indx = Indx(q*(b-1)+a);
end
  
