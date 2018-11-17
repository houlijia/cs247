 function w = myfastdct2_givenT(theta_temp,row,col,I2P,I2Bdct)
 
I_pat_3d               = I2P(reshape(theta_temp,[row, col])); %image2patches3d(I, patch, patch, step, step);
w = I2Bdct(I_pat_3d);

 end
