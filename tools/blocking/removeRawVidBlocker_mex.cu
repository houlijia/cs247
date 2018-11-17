/**
   \file

   MEX function to remove frames into a RawVidBlocker

*/

/**  
  This MEX functions removes all frame preceding \c blk_indx from a RawVidBlkr object

 Input:
   prhs[0] - a uint8 array containing a pointer to the RawVidBlocker object 
   prhs[1] - (uint32) optional temporal index of first block to preserve (sta4ring from 1).
             If not present all frames are removed. 
	     \note In conitnuous operation this will eventually wrap-over, causing
             some problems, but at 30 FPS this will happen after 4.5 years...
             
 Output: None:
*/

#include "mex.h"
#include "matrix.h"

#include "CudaDevInfo.h"

#if HAS_GPU
#include "gpu/mxGPUArray.h"
#endif

#include "mex_tools.h"
  
#if HAS_GPU
#include "mex_gpu_tools.h"
#endif

#include "RawVidBlocker.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  GenericRawVidBlocker *vblkr;

  const char * errId = "removeRawVidBlocker_mex:args_error";

 // Check correctness and get input information
  mex_assert(nlhs == 0 &&  (nrhs == 1 || nrhs == 2), 
	     (errId, "Should have 1 or 2 inputs and no output arguments"));

  // parse prhs[0]
  mex_assert(mxGetClassID(prhs[0]) == mxUINT8_CLASS && 
	     mxGetNumberOfElements(prhs[0]) == sizeof(vblkr),
	     (errId, "1st input should be uint8 array with sizeof(pointer) elements"));
  memcpy(&vblkr, mxGetData(prhs[0]), sizeof(vblkr));

  if(nrhs == 1) {
    vblkr->removeFrms(vblkr->frmsCOUNT());
  }
  else {
    // parse prhs[1]
    mex_assert(mxGetClassID(prhs[1]) == mxUINT32_CLASS && mxIsScalar(prhs[1]),
	       (errId, "2nd input should be uint32 scalar"));

    size_t t_blk = size_t(* (uint32_T *)mxGetData(prhs[1]));

    mex_assert(t_blk > 0,
	       (errId, "2 input argument should be positive"));

    vblkr->removeFrmsBeforeTBlk(t_blk-1); // remove 1 to make 0 start
  }
}
  
