/*
 * main.c
 *
 * Code generation for function 'main'
 *
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/
/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "main2.h"
#include "Nimresize_terminate.h"
#include "Nimresize_emxAPI.h"
#include "Nimresize_initialize.h"
#include <stdio.h>
#include <assert.h>
#include "file_io.h"

/* Function Declarations */
#if 0
static void argInit_1x2_real_T(double result[2]);
static void argInit_480x640x3_uint8_T(unsigned char result[921600]);
static double argInit_real_T(void);
static unsigned char argInit_uint8_T(void);
#endif


/* Function Definitions */
#if 0
static void argInit_1x2_real_T(double result[2])
{
  int b_j1;

  /* Loop over the array to initialize each element. */
  for (b_j1 = 0; b_j1 < 2; b_j1++) {
    /* Set the value of the array element.
       Change this value to the value that the application requires. */
    result[b_j1] = argInit_real_T();
  }
}
#endif

#if 0
static void argInit_480x640x3_uint8_T(unsigned char result[921600])
{
  int b_j0;
  int b_j1;
  int j2;

  /* Loop over the array to initialize each element. */
  for (b_j0 = 0; b_j0 < 480; b_j0++) {
    for (b_j1 = 0; b_j1 < 640; b_j1++) {
      for (j2 = 0; j2 < 3; j2++) {
        /* Set the value of the array element.
           Change this value to the value that the application requires. */
        result[(b_j0 + 480 * b_j1) + 307200 * j2] = argInit_uint8_T();
      }
    }
  }
}
#endif

#if 0
static double argInit_real_T(void)
{
  return 0.0;
}
#endif

#if 0
static unsigned char argInit_uint8_T(void)
{
  return 0;
}
#endif

void main_Nimresize(unsigned char *uv0, const char *outf, unsigned char *outbuf )
{
  emxArray_uint8_T *rout;
  // static unsigned char uv0[921600];
   
  double dv3[2] = { 512.0, 512.0 } ;
  emxInitArray_uint8_T(&rout, 3);

  /* Initialize function 'Nimresize' input arguments. */
  /* Initialize function input argument 'A'. */
  /* Initialize function input argument 'm'. */
  /* Call the entry-point 'Nimresize'. */
  // argInit_480x640x3_uint8_T(uv0);
  // argInit_1x2_real_T(dv3);
  Nimresize(uv0, dv3, rout, outbuf);

  printf("main_Nimresize: outfile %s \n", outf) ;

  printf("rout: alloc %d dim %d can %d size %d %d %d ptr %p \n",
	rout->allocatedSize,
	rout->numDimensions,
	rout->canFreeData,
	rout->size[0],
	rout->size[1],
	rout->size[2],
	rout->data ) ;

  if ( !file_out( outf, rout->size[0] * rout->size[1] * rout->size[2], ( char *)rout->data ))
	  printf("main_Nimresize: file_out failed \n") ;

  emxDestroyArray_uint8_T(rout);
}


#define IN_WID	640
#define IN_HEIGHT	480
#define IN_COLOR	3

// 640 * 480 * 3
#define BUF_SIZE ( IN_WID * IN_HEIGHT * IN_COLOR )

// 512 * 512 * 3
#define OUT_BUF_SIZE 786432

unsigned char *inbuf ;
unsigned char outbuf[ OUT_BUF_SIZE ] ;

#ifdef CUDA_OBS 
unsigned char *orp = NULL, *ogp = NULL, *obp = NULL ;
#endif 

template<typename T>
int
matrix_transpose( T *from, int col, int row )
{
	T *tp, *otp, *fp ;
	int i, j ;

	printf("%s: from %p col %d row %d \n", __func__, (void*)from, col, row ) ;
	
	i = col * row * sizeof ( T ) ;

	otp = tp = ( T *) malloc ( i ) ;
	if ( tp == NULL )
		return ( 0 ) ;

	for ( i = 0 ; i < col ; i++ )
	{
		fp = from + i ;
		for ( j = 0 ; j < row ; j++ )
		{
			*tp++ = *fp ;
			fp += col ;
		}
	}
	
	memcpy ( from, otp, col * row * sizeof ( T )) ; 
	free ( otp ) ;
	return ( 1 ) ;
}

template
int matrix_transpose<int>( int *from, int col, int row ) ;
template
int matrix_transpose<unsigned char>( unsigned char *from, int col, int row ) ;

#ifdef CUDA_OBS 

void
convert_2_rgb( unsigned char *rgbp, int col, int row )
{
	int i ;
	unsigned char *rrp, *ggp, *bbp ;

	if ( orp == NULL )
	{
		orp = ( unsigned char * )malloc ( col * row * sizeof ( unsigned char )) ;
		ogp = ( unsigned char * )malloc ( col * row * sizeof ( unsigned char )) ;
		obp = ( unsigned char * )malloc ( col * row * sizeof ( unsigned char )) ;
	}

	assert ( orp && ogp && obp ) ;

	rrp = orp ;
	ggp = ogp ;
	bbp = obp ;

	i = row * col ;
	while ( i-- )   // BGR ... now
	{
		*bbp++ = *rgbp++ ;
		*ggp++ = *rgbp++ ;
		*rrp++ = *rgbp++ ;
	}
}
#endif 

#ifdef CUDA_OBS 
int main(int argc, const char * const argv[])
{
	int i ;
  (void)argc;
  (void)argv;

  if ( argc < 3 )
  { 
	  printf("usage: %s infile outfile \n", argv[0] );
	  return (0 ) ;
  }

  printf("infile %s outfile %s\n", argv[1], argv[2] ) ;

  inbuf = ( unsigned char *) malloc ( IN_WID * IN_HEIGHT * IN_COLOR * sizeof ( unsigned char )) ;

  if ( inbuf == NULL )
  { 
	  printf("malloc inbuf failed \n", argv[0] );
	  return (0 ) ;
  }

  if ( !file_in( argv[1], BUF_SIZE, inbuf ))
	  return ( 0 ) ;

	convert_2_rgb( inbuf, IN_WID, IN_HEIGHT ) ;

	i = 0 ;
	i += matrix_transpose( obp, IN_WID, IN_HEIGHT ) ;
	i += matrix_transpose( ogp, IN_WID, IN_HEIGHT ) ;
	i += matrix_transpose( orp, IN_WID, IN_HEIGHT ) ;

	if ( i != 3 )
	{
		printf("matrix_transpose failed %d \n", i ) ;
		return ( 0 ) ;
	}

	i = IN_WID * IN_HEIGHT ;

	memcpy( inbuf, obp, i ) ;
	memcpy( inbuf+i, ogp, i ) ;
	memcpy( inbuf+2* i, orp, i ) ;

	// file_out( argv[2], i*3, inbuf) ;

  /* Initialize the application.
     You do not need to do this more than one time. */
  Nimresize_initialize();

  /* Invoke the entry-point functions.
     You can call entry-point functions multiple times. */
  main_Nimresize( inbuf, argv[2] );

  /* Terminate the application.
     You do not need to do this more than one time. */
  Nimresize_terminate();
  return 0;
}
#endif 

/* End of code generation (main.c) */
