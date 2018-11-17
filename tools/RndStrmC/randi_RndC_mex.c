/*
MATLAB function emulating randi(imax,cnt,1):
    function [rnd_state, out] = randi_RndC_mex(rnd_state, imax, cnt)
rnd_state - A RndCState struct
imax - Maximal integer 
cnt - number of random numbers to generates
out - a Mex array of cnt uint32 values
*/

#include <mex.h>
#include <matrix.h>

#include "RndCState_mex.h"
#include "common_RndC_mex.h"
#include "RndC_ifc.h"
#include "RndCState.h"

void mexFunction(int nlhs,
		 mxArray *plhs[],
		 int nrhs,
		 const mxArray *prhs[]
		 ) {
  RndCState rnd_state;
  RndC_uint32 cnt, imax;
  RndC_uint32 *out;
  mxArray *out_mex;
  mwSize dims[2] = {1,1};

  chk_args_cnt_RndC_mex(nlhs, nrhs, "rand_RndC_mex", 2, 3);
  get_RndCState_mex(prhs[0], "rand_RndC_mex:rnd_state", &rnd_state);
  imax = get_uint32_mex(prhs[1], "rand_RndC_mex:imax");
  cnt = get_uint32_mex(prhs[2], "rand_RndC_mex:cnt");
  dims[0] = cnt;

  out_mex = mxCreateNumericArray(2, dims, mxUINT32_CLASS, mxREAL);
  out = mxGetData(out_mex);

  randi_RndC(&rnd_state, imax, cnt, out);

  if (nlhs > 0)
    plhs[0] = RndCState_to_mex(&rnd_state);

  if (nlhs > 1)
    plhs[1] = out_mex;
  else
    mxDestroyArray(out_mex);
}     
	      
  
	     
	     


