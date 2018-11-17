/**
   \file

   MEX function to exctract a pixels vector from a RawvidBlocker object

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

static const char * errId = "getRawVidBlocker_mex:args_error";

static void allocate_output(mwSize vec_len,
			    mxClassID class_id, //*< pixel type class ID
			    bool on_gpu,
			    void *& mx_out,
			    void *& out
			    )
{
#if HAS_GPU
  if(on_gpu) {
    const mwSize dims[] = {vec_len, 1};
    mxGPUArray *mgr = mxGPUCreateGPUArray(2, dims, class_id, mxREAL, 
					  MX_GPU_DO_NOT_INITIALIZE);
    mx_out = mgr;
    out = mxGPUGetData(mgr);
  }
  else {
#endif
    mxArray * mxr = mxCreateUninitNumericMatrix(vec_len, 1, class_id, mxREAL);
    mx_out = mxr;
    out = mxGetData(mxr);
#if HAS_GPU
  }
#endif
}
			    
			  
/**  
  This MEX functions extracts a pixels vector from a RawVidBlkr object. The
  pixel vector may correspond to a single block or to several frame
  blocks. The number of input arguments is 2 or 3 if the request is for a
  single block and 3 or 4 if the request is for several frame blocks. The last
  argument in both cases in \c on_gpu, an optional flag indicating whether the
  output vector is the GPU or on CPU. If not present the vector is in the same
  place as the pixel frames of the RawVidBlocker object.

 Input:
   prhs[0] - a uint8 array containing a pointer to the RawVidBlocker object 
   prhs[1] - blk_id (uint_32). If this is a 3 element vector then the request
             is for a single block and blk_id is the block index (v,h,t). 
             Else blk_id is a scalar, in which case the request is for several
             frame blocks beginning at frame block blk_id. In both cases
             indices start from 1 and temporal index is absolute.
   prhs[2] - if the request is for a single block this field is the optional
             \c on_gpu (logical scalar). Otherwise, this is the number of
             frame blocks to be read (uint_32).
   prhs[3] - Optional \c on_gpu for the case that a number of blocks is requested.

             
 Output:
   plhs[0] - the pixels vector
*/
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  GenericRawVidBlocker *vblkr;

  // Check number of arguments
  mex_assert(nlhs == 1 &&  (nrhs >= 2 || nrhs <= 4), 
	     (errId, "Should have 2 to 4 inputs and 1 output arguments"));

  // parse prhs[0]
  mex_assert(mxGetClassID(prhs[0]) == mxUINT8_CLASS && 
	     mxGetNumberOfElements(prhs[0]) == sizeof(vblkr),
	     (errId, "1st input should be uint8 array with sizeof(pointer) elements"));
  memcpy(&vblkr, mxGetData(prhs[0]), sizeof(vblkr));

  // parse prhs[1]
  mex_assert(mxGetClassID(prhs[1]) == mxUINT32_CLASS,
	     (errId, "2nd input should be a uint32"));

  bool single_blk = !mxIsScalar(prhs[1]);

  // parse on_gpu
  int on_gpu_indx = single_blk? 2: 3;
  bool on_gpu;
  if(on_gpu_indx < nrhs) {
    mex_assert(mxIsLogicalScalar(prhs[on_gpu_indx]),
	       (errId, "on_gpu argument must be a logical scalar"));
    on_gpu = mxIsLogicalScalarTrue(prhs[on_gpu_indx]);
#if !HAS_GPU
    mex_assert(!on_gpu,
	       (errId, "on_gpu argument must be false since there is no GPU"));
#endif
  }
  else
    on_gpu = vblkr->onGPU();	// Default

  // Parse blk_id
  const uint32_T * pblk_id = (const uint32_T *) mxGetData(prhs[1]);
  mwSize vec_len;
  void *mx_output;
  void *output;
  
  if(single_blk) {
    mex_assert(mxGetNumberOfElements(prhs[1]) == 3, 
	       (errId, "In single block mode, 2nd argument should have 3 elements"));
    mex_assert((nrhs == 2 || nrhs == 3), 
	     (errId, "In single block mode, Should have 2 or 3 inputs"));
    unsigned blk_v = unsigned(pblk_id[0]);
    unsigned blk_h = unsigned(pblk_id[1]);
    size_t blk_t = size_t(pblk_id[2]);
    mex_assert(blk_v > 0 && blk_h > 0 && blk_t > 0,
	       (errId, "blk_v, blk_h, blk_t should be positive"));
    blk_v--; blk_h--; blk_t--;

    // Allocate output
    vec_len = (mwSize) vblkr->blkLENGTH(blk_v, blk_h);
    allocate_output(vec_len, vblkr->pixelCLASSID(), on_gpu, mx_output, output);
    
    // extract output
    vblkr->getBlk(blk_v, blk_h, blk_t, output, NULL, on_gpu);
  }
  else {
    mex_assert((nrhs == 3 || nrhs == 4), 
	     (errId, "In frames blocks mode, Should have 3 or 4 inputs"));
    unsigned frm_blk_0 = unsigned(*pblk_id);
    mex_assert(frm_blk_0>0, 
	       (errId, "block_id should be positive"));
    frm_blk_0--;

    // parse prhs[2]
    mex_assert(mxIsScalar(prhs[2]) && mxGetClassID(prhs[2]) == mxUINT32_CLASS,
	       (errId, "3rd argument (number of frame blocks) should be scalar uint_32"));
    unsigned n_frm_blks = *(uint32_T *) mxGetData(prhs[2]);

    // Allocate output
    vec_len = (mwSize) (n_frm_blks * vblkr->frmBlkLENGTH());
    allocate_output(vec_len, vblkr->pixelCLASSID(), on_gpu, mx_output, output);

    // extract output
    vblkr->getFrmBlks(n_frm_blks, frm_blk_0, output, NULL, on_gpu);    
  }

  // set output
 #if HAS_GPU
  if(on_gpu) {
    mxGPUArray *mgr = static_cast<mxGPUArray*>(mx_output);
    plhs[0] = mxGPUCreateMxArrayOnGPU(mgr);
    mxGPUDestroyGPUArray(mgr);
  }
  else
#endif
    plhs[0] = static_cast<mxArray*>(mx_output);
}  
 
