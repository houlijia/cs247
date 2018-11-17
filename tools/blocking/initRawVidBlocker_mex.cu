/**
   \file

   MEX function to create and initialize a RawVidBlkr object

*/

/**  
  This MEX functions creates and initializes a RawVidBlkr object

 Input:
   prhs[0] - A cell array containing the first frame. The function saves this frame
             and in addtion extracts frame specification using this frame as an
             example. Frames are stored on GPU if this example frame is on GPU. 
   prhs[1] - an array of type uint16 and size [n_clr, 3], where n_clr is the
             number of color components and edeach row [v,h,t] contains the
             [vertical, horizontal, temporal] dimension of a raw video
             block. Note that the temporal components are assumed to be identical.
   prhs[2] - an array of type uint16 and size [n_clr, 3], where n_clr is the
             number of color components and each row [v,h,t] contains the
             [vertical, horizontal, temporal] overlap between raw video
             blocks. Note that the temporal components are assumed to be identical.
             
 Output:
   plhs[0] - A uint8 array containing the address of the generated RawVidBlkr.

*/

#include <string.h>

#include "mex.h"
#include "matrix.h"

#include "mex_assert.h"
#include "CudaDevInfo.h"

#if HAS_GPU
#include "gpu/mxGPUArray.h"
#endif

#include "mex_tools.h"
  
#if HAS_GPU
#include "mex_gpu_tools.h"
#endif

#include "RawVidBlocker.h"

static const char * errId = "initRawVidBlocker_mex:args_error";

template<typename alloc>
GenericRawVidBlocker *newVidBlocker(size_t n_clr,
				    size_t dm[][2],
				    size_t b_s_sz[][2],
				    size_t b_s_olp[][2],
				    size_t b_t_sz, 
				    size_t b_t_olp
				    )
{
  VidFrameSpec<alloc> spec(n_clr, dm);
  return new RawVidBlocker<alloc>(spec, b_s_sz, b_s_olp, b_t_sz, b_t_olp);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  unsigned c,d;
  const uint16_T *p, *q;
  mxClassID pixel_t_id;
  GenericRawVidBlocker *vblkr;

  // Check correctness and get input information
  mex_assert(nlhs == 1 &&  nrhs == 3, 
	     (errId, "Should have 3 input and 1 output arguments"));

  mex_assert(mxIsCell(prhs[0]),
             (errId, "1st argument should be a cell array"));

  size_t n_clr = mxGetNumberOfElements(prhs[0]);
  mex_assert(n_clr > 0 && n_clr<=VidSpec::max_colors, 
	    (errId, "number of colors is 0 or too large"));

  mxArray *mx_clr = mxGetCell(prhs[0], 0);

  size_t dm[VidSpec::max_colors][2];
  const void *vid[VidSpec::max_colors];

#if HAS_GPU
  
  bool on_gpu = mxIsGPUArray(mx_clr);

  const mxGPUArray *pg[VidSpec::max_colors];
  if(on_gpu) {
    pg[0] = mxGPUCreateFromMxArray(mx_clr);

    pixel_t_id = mxGPUGetClassID(pg[0]);

    mex_assert(mxGPUGetNumberOfDimensions(pg[0]) == 2, 
	       (errId, "frame should have 2 dims"));

    const mwSize * pdim = mxGPUGetDimensions(pg[0]);
    dm[0][0] = size_t(pdim[0]);
    dm[0][1] = size_t(pdim[1]);
    mxFree((void *)pdim);
    vid[0] = mxGPUGetDataReadOnly(pg[0]);

    for(c=1; c<n_clr; c++) {
      mx_clr = mxGetCell(prhs[0], c);
      mex_assert(mxIsGPUArray(mx_clr) == on_gpu,
		 (errId, "All color frames should be in same memory"));

      pg[c] = mxGPUCreateFromMxArray(mx_clr);
      
      mex_assert(mxGPUGetClassID(pg[c]) == pixel_t_id, 
		 (errId, "All frames should be of the same type"));
      mex_assert(mxGPUGetNumberOfDimensions(pg[c]) == 2, 
		 (errId, "frame should have 2 dims"));

      pdim = mxGPUGetDimensions(pg[c]);      
      dm[c][0] = size_t(pdim[0]);
      dm[c][1] = size_t(pdim[1]);
      mxFree((void *)pdim);
      
      vid[c] = mxGPUGetDataReadOnly(pg[c]);
    }
  }
  else {    
#endif

    pixel_t_id = mxGetClassID(mx_clr);

    mex_assert(mxGetNumberOfDimensions(mx_clr) == 2, 
               (errId, "frame should have 2 dims"));
	       
    dm[0][0] = mxGetM(mx_clr);
    dm[0][1] = mxGetN(mx_clr);
    vid[0] = mxGetData(mx_clr);

    for(c=1; c<n_clr; c++) {
      mx_clr = mxGetCell(prhs[0], c);

#if HAS_GPU      
      mex_assert(mxIsGPUArray(mx_clr) == on_gpu,
		 (errId, "All color frames should be in same memory"));
#endif

      mex_assert(mxGetNumberOfDimensions(mx_clr) == 2, 
		 (errId, "frame should have 2 dims"));
      mex_assert(mxGetClassID(mx_clr) == pixel_t_id, 
                 (errId, "All frames should be of the same type"));
      
      dm[c][0] = mxGetM(mx_clr);
      dm[c][1] = mxGetN(mx_clr);
      vid[c] = mxGetData(mx_clr);
    }
#if HAS_GPU
  }
#endif
  
  size_t b_s_sz[VidSpec::max_colors][2];
  size_t b_s_olp[VidSpec::max_colors][2];
  size_t b_t_sz, b_t_olp;

  p = (uint16_T *)mxGetData(prhs[1]);
  q = (uint16_T *)mxGetData(prhs[2]);
  for(d=0; d<2; d++)
    for(c=0; c<n_clr; c++) {
      b_s_sz[c][d] = *p++;
      b_s_olp[c][d] = *q++;
    }

  b_t_sz = *p; b_t_olp = *q;

#if HAS_GPU
  if(on_gpu) {
    switch(pixel_t_id) {
    case mxDOUBLE_CLASS:
      vblkr = newVidBlocker<D_RectAlloc<double> >(n_clr, dm, b_s_sz, b_s_olp, 
						b_t_sz, b_t_olp);
      break;
    case mxSINGLE_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<float> >(n_clr, dm, b_s_sz, b_s_olp, 
					       b_t_sz, b_t_olp);
      break;
    case mxUINT8_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<uint8_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						 b_t_sz, b_t_olp);
      break;
    case mxUINT16_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<uint16_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						  b_t_sz, b_t_olp);
      break;
    case mxUINT32_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<uint32_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						  b_t_sz, b_t_olp);
      break;
    case mxINT8_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<int8_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						b_t_sz, b_t_olp);
      break;
    case mxINT16_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<int16_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						 b_t_sz, b_t_olp);
      break;
    case mxINT32_CLASS: 
      vblkr = newVidBlocker<D_RectAlloc<int32_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						 b_t_sz, b_t_olp);
      break;
    
    default:
      mexPrintf(" **** class ID for pixels: %d ****\n", int(pixel_t_id));
      mexErrMsgIdAndTxt (errId, "Unexpected class for pixels");
    
    }

  }
  else {
    switch(pixel_t_id) {
    case mxDOUBLE_CLASS:
      vblkr = newVidBlocker<H_RectAlloc<double> >(n_clr, dm, b_s_sz, b_s_olp, 
						b_t_sz, b_t_olp);
      break;
    case mxSINGLE_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<float> >(n_clr, dm, b_s_sz, b_s_olp, 
					       b_t_sz, b_t_olp);
      break;
    case mxUINT8_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<uint8_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						 b_t_sz, b_t_olp);
      break;
    case mxUINT16_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<uint16_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						  b_t_sz, b_t_olp);
      break;
    case mxUINT32_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<uint32_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						  b_t_sz, b_t_olp);
      break;
    case mxINT8_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<int8_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						b_t_sz, b_t_olp);
      break;
    case mxINT16_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<int16_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						 b_t_sz, b_t_olp);
      break;
    case mxINT32_CLASS: 
      vblkr = newVidBlocker<H_RectAlloc<int32_T> >(n_clr, dm, b_s_sz, b_s_olp, 
						 b_t_sz, b_t_olp);
      break;
    
    default: 
      mexPrintf(" **** class ID for pixels: %d ****\n", int(pixel_t_id));
      mexErrMsgIdAndTxt (errId, "Unexpected class for pixels");
    }
  }

  vblkr->insertFWD(1, vid, on_gpu);

  // Destroy the pg array, if created on GPU
  if(on_gpu)
    for(c=0; c<n_clr; c++) 
      mxGPUDestroyGPUArray(pg[c]);


#else
  switch(pixel_t_id) {
  case mxDOUBLE_CLASS:
    vblkr = newVidBlocker<RectAlloc<double> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxSINGLE_CLASS: 
    vblkr = newVidBlocker<RectAlloc<float> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxUINT8_CLASS: 
    vblkr = newVidBlocker<RectAlloc<uint8_T> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxUINT16_CLASS: 
    vblkr = newVidBlocker<RectAlloc<uint16_T> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxUINT32_CLASS: 
    vblkr = newVidBlocker<RectAlloc<uint32_T> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxINT8_CLASS: 
    vblkr = newVidBlocker<RectAlloc<int8_T> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxINT16_CLASS: 
    vblkr = newVidBlocker<RectAlloc<int16_T> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
  case mxINT32_CLASS: 
    vblkr = newVidBlocker<RectAlloc<int32_T> >(n_clr, dm, b_s_sz, b_s_olp, 
					      b_t_sz, b_t_olp);
    break;
    
  default:
    mexPrintf(" **** class ID for pixels: %d ****\n", int(pixel_t_id));
    mexErrMsgIdAndTxt (errId, "Unexpected class for pixels");
  }

  vblkr->insertFWD(1, vid);

#endif


 // Create output
  plhs[0] = mxCreateNumericMatrix(1, sizeof(vblkr), mxUINT8_CLASS, mxREAL);
  mex_assert((plhs[0] != NULL), (errId, 
				 "Failed to allocate %lu uint8 for output",
				 (unsigned long) sizeof(vblkr)));

  memcpy(mxGetData(plhs[0]), &vblkr, sizeof(vblkr));
}
