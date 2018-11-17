 function  A = A_wht_ord(z, Mea_ind)
   
temp = myfwht(z);
% temp =fwht(z(:));
 A = squeeze(temp(Mea_ind));
 
end