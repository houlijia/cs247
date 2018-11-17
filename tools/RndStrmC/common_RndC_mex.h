#ifndef common_RndC_mex_h
#define common_RndC_mex_h

#include "RndC_ifc.h"

void
chk_args_cnt_RndC_mex(int nlhs, /* no. of output args */
		     int nrhs, /* no. of input args */\
		     const char *name, /* function name */
		     int max_nlhs, /* max no. of output args */
		     int xpct_nrhs /* Expected no. of input args */
		     );

/* Convert a mxArray containing a scalar numeric int a uint32, verifying
   that this is the correct value
*/
RndC_uint32
get_uint32_mex(const mxArray *mx_val, /* value to convert */
	       const char *name	      /* Variable name */
	       );

/* Read a RndCState struct from a scalar mxArray containing a struct */
void 
get_RndCState_mex(const mxArray *mx_val, /* value to convert */
		  const char *name,	      /* Variable name */
		  struct RndCState *rnd_state /* Converted value */
		  );

#endif	/* common_RndC_mex_h */
