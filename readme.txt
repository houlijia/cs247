
This is the MATLAB code for CS vs. JPEG.
We have developed a new image compression system based on compressive sensing.
Reported in the paper:
Xin Yuan, Raziel Haimi-Cohen and Paul A. Wilford
"Image Compression Based on Compressive Sensing: End-to-End Comparison with JPEG",
Submitted to IEEE Transactions on Image Processing


It inlcudes the encoder and decoder.
The following algorithms are included for image reconstruction
1) GAP-TV 
X. Yuan. "Generalized alternating projection based total variation minimization for compressive sensing." In 2016 IEEE International Conference on Image Processing (ICIP), pages 2539–2543, Sept 2016.
Code downloaded from:
https://sites.google.com/site/eiexyuan/publications

2) D-AMP
C. A. Metzler, A. Maleki and R. G. Baraniuk, "From Denoising to Compressed Sensing," in IEEE Transactions on Information Theory, vol. 62, no. 9, pp. 5117-5144, Sept. 2016.
Code downloaded from
https://dsp.rice.edu/software/DAMP-toolbox

3) NLR-CS
W. Dong, G. Shi, X. Li, Y. Ma, and F. Huang. "Compressive sensing via nonlocal low-rank regularization". IEEE Transactions on Image Processing, 23(8):3618–3632, 2014.
Code downloaded from:
http://see.xidian.edu.cn/faculty/wsdong/

Please first run
-- demo_firstTry.m
   to get a first impression on what you will see
Then run
-- demo_Figure1.m
   to reproduce the results in Figure 1 of the paper

After that, please run
-- demo_GAPTV.m
   You will see the results of 8 images using sensing matrix 2D-DCT and 2D-WHT and the reconstruction algorithm GAP-TV
   After this, you can get the encoded compressive measurements for each image at
./test_data/Image and a folder with the filename

Following this, you can run
-- demo_read_CSfile_and_Reconstruct.m
   You will read the CS files and reconstruct the original images using different reconstruction algorithms.
   You can reproduce the results in Figure 4 of the paper after this demo is done
   How to use other algorithms are also introduced in this demo

If you want to get Figure 4 directly, you can simply run
-- demo_Figure4.m

If you want to get the results using SRM-WHT and SRM-DCT as sensing matrix
please run
-- demo_GAPTV_SRM.m
-- demo_Read_CSfile_and_Reconstruct_SRM.m

If you want to compare the results with and without quantization, please run
-- demo_CompQuan_GAPTV.m
-- demo_CompQuan_3Algo.m


Comments please kindly send to
Xin Yuan & Raziel Haimi-Cohen
Nokia Bell Labs
xyuan@bell-labs.com;  eiexyuan@gmail.com; razihc@gmail.com 
     
