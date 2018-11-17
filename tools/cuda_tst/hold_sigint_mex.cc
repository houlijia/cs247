/** \file */

#include <stdio.h>
#include <unistd.h>
#include "mex.h"

#include "hold_sigint.h"
#include "mex_assert.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  (void) prhs;
  (void) plhs;
  (void) nrhs;
  (void) nlhs;

  mex_assert(nlhs==1 && nrhs==0,
	     ("hold_sigint_mex:args",
	      "hold_sigint_mex should have no inputs and one output"));

  for(int k=5; k>0; --k) {
    mexPrintf("Sleeping...\n");
    mexEvalString("drawnow;");
    sleep(2);
  }
  printf("Done\n");
}
		 
