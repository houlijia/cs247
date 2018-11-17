function  At = At_wht_ord(z, Mea_ind,row, col)
   
 len = row*col; 
 z_all = zeros(len,1);
 z_all(Mea_ind) = z;
 %z_all = [z; zeros(len-Mea_num,1)];
 At = myfwht(z_all);
 At= At./(row*col);
% A = ifwht(z_all);
 %A = temp(1:Mea_num);
 
end