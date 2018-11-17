/*
 * example.c
 *
 * This file illustrates how to use the IJG code as a subroutine library
 * to read or write JPEG image files.  You should look at this code in
 * conjunction with the documentation file libjpeg.doc.
 *
 * This code will not do anything useful as-is, but it may be helpful as a
 * skeleton for constructing routines that call the JPEG library.  
 *
 * We present these routines in the same coding style used in the JPEG code
 * (ANSI function definitions, etc); but you are of course free to code your
 * routines in a different style if you prefer.
 */

#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rgb2jpg.h"

/*
 * Include file for users of JPEG library.
 * You will need to have included system headers that define at least
 * the typedefs FILE and size_t before you can include jpeglib.h.
 * (stdio.h is sufficient on ANSI-conforming systems.)
 * You may also wish to include "jerror.h".
 */

#include "jpeglib.h"

/*
 * <setjmp.h> is used for the optional error recovery mechanism shown in
 * the second part of the example.
 */

#include <setjmp.h>

typedef struct jpeg_compress_struct jpeg_compress_struct;

/* setup the buffer but we did that in the main function */
void init_buffer(jpeg_compress_struct* cinfo) {}

/* what to do when the buffer is full; this should almost never
* happen since we allocated our buffer to be big to start with
*/
boolean empty_buffer(jpeg_compress_struct* cinfo) {
	return TRUE;
}

/* finalize the buffer and do any cleanup stuff */
void term_buffer(jpeg_compress_struct* cinfo) {}

unsigned char *
rgb2jpg( unsigned char *rgbp, int col, int row, int *size ) 
{
	/* load the image and note I assume this is a RGB 24 bit image
	* you should do more checking/conversion if possible
	*/

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr       jerr;
	struct jpeg_destination_mgr dmgr;

	/* create our in-memory output buffer to hold the jpeg */
	JOCTET * out_buffer   = new JOCTET[row * col * 3];

	/* here is the magic */
	dmgr.init_destination    = init_buffer;
	dmgr.empty_output_buffer = empty_buffer;
	dmgr.term_destination    = term_buffer;
	dmgr.next_output_byte    = out_buffer;
	dmgr.free_in_buffer      = col * row * 3 ; 

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	/* make sure we tell it about our manager */
	cinfo.dest = &dmgr;

	cinfo.image_width      = col ;
	cinfo.image_height     = row ;
	cinfo.input_components = 3;
	cinfo.in_color_space   = JCS_RGB;
	// cinfo.next_scanline   = 0;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality (&cinfo, 75, true);
	jpeg_start_compress(&cinfo, true);

	JSAMPROW row_pointer;
	unsigned char *buffer    = rgbp ;

#ifdef CUDA_OBS 
	/* silly bit of code to get the RGB in the correct order */
	for (int x = 0; x < col; x++) {
		for (int y = 0; y < row ; y++) {
			unsigned char *p = rgbp + y * row * 3 + x * 3 ;
			unsigned char c = *p ;
			*p = *(p+2) ;
			*(p+2) = c ;
				// swap (p[0], p[2]);
		}
	}
#endif 

	/* main code to write jpeg data */
	while (cinfo.next_scanline < cinfo.image_height) { 		
		// row_pointer = (JSAMPROW) &buffer[cinfo.next_scanline * image->pitch];
		row_pointer = (JSAMPROW) &buffer[cinfo.next_scanline * col * 3 ];
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	}
	jpeg_finish_compress(&cinfo);

	/* write the buffer to disk so you can see the image */
	*size = cinfo.dest->next_output_byte - out_buffer ;
	// fwrite(out_buffer, cinfo.dest->next_output_byte - out_buffer,1 ,outfile);

  	// ojpeg_destroy_compress(&cinfo);
	return ( out_buffer ) ;
}
