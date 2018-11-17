#include <stdio.h>
#include <stdlib.h>

unsigned char *
do_bicubic(unsigned char *data, int width, int height, int newWidth, int newHeight)
{
	unsigned char * Data = ( unsigned char *)malloc ( newWidth * newHeight * 3 ) ;
	unsigned char  Cc;
	unsigned char  C[5];
	unsigned char  d0,d2,d3,a0,a1,a2,a3;
	int i,j,k,jj;
	int x,y;
	float dx,dy;
	float tx,ty;
	int nChannels, widthStep ;

	tx = (float)width/(float)newWidth ;
	ty =  (float)height / ( float )newHeight;
	printf("new %d %d old %d %d ratio %f %f", newWidth, newHeight, width, height, tx, ty ) ;

	if ( Data == NULL )
	{
		printf("%s :: malloc failed \n", __func__ ) ;
		return ( Data ) ;
	}

	nChannels = 1 ;
	widthStep = width ;

	for(i=0; i<newHeight; i++)
	{
		for(j=0; j<newWidth; j++)
		{
			printf("%d : %d\n",i,j);

			x = (int)(tx*j);
			y =(int)(ty*i);

			dx= tx*j-x;
			dy=ty*i -y;

			for(k=0;k<3;k++)
			{
				for(jj=0;jj<=3;jj++)
				{

					d0 = data[(y-1+jj)*widthStep + (x-1)*nChannels +k] - data[(y-1+jj)*widthStep + (x)*nChannels +k] ;
					d2 = data[(y-1+jj)*widthStep + (x+1)*nChannels +k] - data[(y-1+jj)*widthStep + (x)*nChannels +k] ;
					d3 = data[(y-1+jj)*widthStep + (x+2)*nChannels +k] - data[(y-1+jj)*widthStep + (x)*nChannels +k] ;
					a0 = data[(y-1+jj)*widthStep + (x)*nChannels +k];
					a1 =  -1.0/3*d0 + d2 -1.0/6*d3;
					a2 = 1.0/2*d0 + 1.0/2*d2;
					a3 = -1.0/6*d0 - 1.0/2*d2 + 1.0/6*d3;
					C[jj] = a0 + a1*dx + a2*dx*dx + a3*dx*dx*dx;

					d0 = C[0]-C[1];
					d2 = C[2]-C[1];
					d3 = C[3]-C[1];
					a0=C[1];
					a1 =  -1.0/3*d0 + d2 -1.0/6*d3;
					a2 = 1.0/2*d0 + 1.0/2*d2;
					a3 = -1.0/6*d0 - 1.0/2*d2 + 1.0/6*d3;
					Cc = a0 + a1*dy + a2*dy*dy + a3*dy*dy*dy;
					// if((int)Cc>255) Cc=255;
					//                 if((int)Cc<0) Cc=0;
					Data[i*widthStep +j*nChannels +k ] = Cc;
				}
			}
		}
	}
	return Data ;   
}
