// Tue Oct 27 15:03:53 EDT 2015

For instance, you can run

>> I = imread(‘XXX.jpg’);
>> [CM,J] = LocalizedOrderedSensing(I,128);

Then you have compressed measurements as CM.

J is a resized original image with size 512x512 at p = 128 or 256x256 at p = 64.

'p' s a kind of parameter that sets the resolution of the resized image.
If p = 128, the size of the returned image ‘J’ is 512x512.

The resized image is actually considered as the original image to be compressively measured.
CM is a cell type variable.
It has 6 cells. Each has 128x128 measurements at p=128.

-------------------------------------------------------------

figure
imshow(I)
title('Original Image')
figure
imshow(J)
title('Resized Image')
