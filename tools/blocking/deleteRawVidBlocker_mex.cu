/**
   \file

   MEX function to delete a RawVidBlkr object

*/

/**  
  This MEX functions deletes a RawVidBlkr object

 Input:
   prhs[0] - an pointer to the object (as an array of bytes).
             
 Output: NONE
   

*/

#include "mex.h"
#include "matrix.h"

#include "mex_assert.h"
#include "RawVidBlocker.h"

static const char * errId = "deleteRawVidBlocker_mex:args_error";

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  GenericRawVidBlocker *vblkr;

  mex_assert(nlhs == 0 && nrhs == 1, 
	     (errId, "Should have 1 input and 0 output arguments"));

  mex_assert(mxGetClassID(prhs[0]) == mx_class_id<uint8_T>(),
	     (errId, "1st argument should be uint16\n"));

  mex_assert(mxGetNumberOfElements(prhs[0]) == sizeof(vblkr),
	     (errId, "Number of elements in 1st argument should "
			 "be sizeof(GenericRawVidBlocker *)\n"));

  memcpy(&vblkr, mxGetData(prhs[0]), sizeof(vblkr));

  delete vblkr;
}

  
