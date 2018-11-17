/*
MATLAB function emulating rand(cnt,1):
    function [rnd_state, out] = rand_RndC_mex(rnd_state, cnt)
rnd_state - A RndCState struct
cnt - number of random numbers to generates
out - a Mex array of cnt doubles
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
  RndC_uint32 cnt;
  double *out;
  mxArray *out_mex;

  chk_args_cnt_RndC_mex(nlhs, nrhs, "rand_RndC_mex", 2, 2);
  get_RndCState_mex(prhs[0], "rand_RndC_mex:rnd_state", &rnd_state);
  cnt = get_uint32_mex(prhs[1], "rand_RndC_mex:cnt");

  out_mex = mxCreateDoubleMatrix(cnt, 1, mxREAL);
  out = mxGetPr(out_mex);

  rand_RndC(&rnd_state, cnt, out);

  if (nlhs > 0)
    plhs[0] = RndCState_to_mex(&rnd_state);

  if (nlhs > 1)
    plhs[1] = out_mex;
  else
    mxDestroyArray(out_mex);
}     
	      
  
	     
	     


