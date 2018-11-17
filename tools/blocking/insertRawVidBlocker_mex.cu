/**
   \file

   MEX function to insert frames into a RawVidBlocker

*/

/**  
  This MEX functions inserts frames into a RawVidBlocker Object

 Input:
   prhs[0] - a uint8 array containing a pointer to the RawVidBlocker object 
   prhs[1] - a uint32 scalar - number of frames
   prhs[2] - a cell array containing the frames (each cell contains one color)
   prhs[3] - Optional logical scalar. If true (default) insertion is forward, else
             backward.
             
 Output: None:
*/

#include "mex.h"
#include "matrix.h"

#include "CudaDevInfo.h"
#include "mex_tools.h"
  
#if HAS_GPU
#include "mex_gpu_tools.h"
#endif

#include "RawVidBlocker.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  GenericRawVidBlocker *vblkr;
  unsigned nfr;
  const void *vid[VidSpec::max_colors];
  bool fwd = true;
  mxArray *mx_clr;
  unsigned c;
  const char * errId = "insertRawVidBlocker_mex:args_error";
	
 // Check correctness and get input information
  mex_assert(nlhs == 0 &&  (nrhs == 3 || nrhs == 4), 
	     (errId, "Should have 3 or 4 inputs and no output arguments"));

  // parse prhs[0]
  mex_assert(mxGetClassID(prhs[0]) == mxUINT8_CLASS && 
	     mxGetNumberOfElements(prhs[0]) == sizeof(vblkr),
	     (errId, "1st input should be uint8 array with sizeof(pointer) elements"));
  memcpy(&vblkr, mxGetData(prhs[0]), sizeof(vblkr));
  
  // parse prhs[1]
  mex_assert(mxGetClassID(prhs[1]) == mxUINT32_CLASS && mxIsScalar(prhs[1]),
	     (errId, "2nd input should be a scalar uint32"));

  nfr = unsigned(*(uint32_T*)(mxGetData(prhs[1])));

  // parse prhs[2]
  mex_assert(mxIsCell(prhs[2]) && mxGetNumberOfElements(prhs[2]) <= vblkr->nColors(), 
	     (errId, "2nd input argument should be a cell array with nColors() cells"));

#if HAS_GPU
  const mxGPUArray *pg[VidSpec::max_colors];
#endif

  for(c=0; c<mxGetNumberOfElements(prhs[2]); c++) {
    mx_clr = mxGetCell(prhs[2], c);

#if HAS_GPU
    mex_assert(mxIsGPUArray(mx_clr) == vblkr->onGPU(),
	       (errId,
		"All color frames should be in same memory.vblkr->onGPU()=%d but clr[%u].on_gpu=%d",
		int(vblkr->onGPU()), c, int(mxIsGPUArray(mx_clr))));

    if(vblkr->onGPU()) {
      pg[c] = mxGPUCreateFromMxArray(mx_clr);

      mex_assert(mxGPUGetClassID(pg[c]) == vblkr->pixelClassID(), 
		 (errId, "%s:%d clr %u: frame color type %u expected %u\n",
		  __FILE__, __LINE__, c,  
		  (unsigned) mxGPUGetClassID(pg[c]), (unsigned) vblkr->pixelClassID()));
      mex_assert(mxGPUEqualMatDims(pg[c],  vblkr->lengthVFRM(c), vblkr->lengthHFRM(c)),
		 (errId, "%s:%d clr %u: %u dimensions. dimension [%u,%u] expected [%u,%u]",
		  __FILE__, __LINE__, c, mxGPUGetNumberOfDimensions(pg[c]),
		  (unsigned) mxGPUGetDim(pg[c],0), (unsigned) mxGPUGetDim(pg[c],1), 
		  (unsigned) vblkr->lengthVFRM(c), (unsigned) vblkr->lengthHFRM(c)));
      vid[c] = mxGPUGetDataReadOnly(pg[c]);
    }
    else {
#endif

      mex_assert(mxGetClassID(mx_clr) == vblkr->pixelClassID(),
		 (errId, "%s:%d clr %u: frame color type %u expected %u\n",
		  __FILE__, __LINE__, c,
		  (unsigned) mxGetClassID(mx_clr),  (unsigned) vblkr->pixelClassID()));
      
      const mwSize *pdim = mxGetDimensions(mx_clr);
      (void) pdim;

      mex_assert(pdim[0] == vblkr->lengthVFRM(c) &&  pdim[1] == vblkr->lengthHFRM(c),
		 (errId, "%s:%d clr %u: dimension [%u,%u] expected [%u,%u]",
		  __FILE__, __LINE__, c,
		  (unsigned) pdim[0], (unsigned) pdim[1],
		  (unsigned) vblkr->lengthVFRM(c), (unsigned) vblkr->lengthHFRM(c)));
      
      vid[c] = mxGetData(mx_clr);
      
#if HAS_GPU
    }
#endif
  }
   
 // parse prhs[3] Running only partial checking
  if(nrhs == 4) {
    mex_assert(mxIsLogicalScalar(prhs[3]),
               (errId, "4th argument must be a logical scalar"));

    fwd = mxIsLogicalScalarTrue(prhs[3]);
  }

  if(fwd)
    vblkr->insertFWD(nfr, vid, vblkr->onGPU());
  else
    vblkr->insertBWD(nfr, vid, vblkr->onGPU());

  
#if HAS_GPU
  // Destroy the pg array, if created on GPU
  if(vblkr->onGPU())
    for(c=0; c<vblkr->nColors(); c++) 
      mxGPUDestroyGPUArray(pg[c]);
#endif

}
    
