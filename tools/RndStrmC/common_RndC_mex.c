#include <stdio.h>
#include <mex.h>

#include "common_RndC_mex.h"
#include "RndCState_mex.h"

void 
chk_args_cnt_RndC_mex(int nlhs,
		      int nrhs,
		      const char *name,
		      int max_nlhs,
		      int xpct_nrhs
		      )
{
  char err_msg[256];

  if (nlhs > max_nlhs) {
    sprintf(err_msg, "%s: %d output arguments (max %d)",
	    name, nlhs, max_nlhs);
    mexErrMsgTxt(err_msg);
  }
  if (nrhs != xpct_nrhs) {
    sprintf(err_msg, "%s: %d input arguments (expected %d)",
	    name, nrhs, xpct_nrhs);
    mexErrMsgTxt(err_msg);
  }
}  

RndC_uint32
get_uint32_mex(const mxArray *mx_val,
	       const char *name
	       )
{
  double dval;
  RndC_uint32 val;
  char err_msg[256];

  if (!mxIsNumeric(mx_val) || mxGetNumberOfDimensions(mx_val) != 2 ||
      mxGetM(mx_val) != 1 ||  mxGetN(mx_val) != 1){
    sprintf(err_msg, "%s must be a numeric scalar", name);
    mexErrMsgTxt(err_msg);
  }

  dval = mxGetScalar(mx_val);
  val = (RndC_uint32)dval;
  if ((double)val != val) {
    sprintf(err_msg,
	    "%s=%f - must be a non-negative integer in uint32 range",
	    name, dval);
  }

  return val;

}

void 
get_RndCState_mex(const mxArray *mx_val, 
		  const char *name,
		  struct RndCState *rnd_state
		  )
{
  char err_msg[256];

  if (!mxIsStruct(mx_val) || mxGetNumberOfDimensions(mx_val) != 2 ||
      mxGetM(mx_val) != 1 ||  mxGetN(mx_val) != 1) {
    sprintf(err_msg, "%s should be a [1 1] struct", name);
    mexErrMsgTxt(err_msg);
  }

  RndCSTate_from_mex(rnd_state, mx_val);

}
