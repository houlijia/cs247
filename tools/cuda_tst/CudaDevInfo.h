#ifndef __CudaDevInfo_HDR__
#define __CudaDevInfo_HDR__

#include <stdio.h>
#include <assert.h>

#ifdef __NVCC__

#include <stdlib.h>

#include "builtin_types.h"
#include "GlobalPtr.h"

class GPUInfo {
public:
  //* Initializes the GPU using the current device.
  GPUInfo()
  { init(getDeviceIndex()); }

  //* Set the device to \c indx.
  void  setDeviceIndex(int indx)
  { if(indx != deviceIndex()) init(indx); }

  //* Get the current device index from the system
  static int getDeviceIndex();

  //* Get the device index as saved in GPUInfo
  int deviceIndex() const
  { return dev_indx; }

  //* Get number of devices
  static int deviceCount();
  
  const struct cudaDeviceProp &getProp() const
  { return prop; }

  //* Returns log2 of the maximal power of 2 not exceeding prop.maxThreadsPerBlock
  int log2ThrdsBlk() const
  { return log2_thrds_blk; }

private:
  void init(int dev_ind);

  int dev_indx;		   //!< Index of current device (-1 undefined)
  struct cudaDeviceProp prop;	//!< NVidia struct of device properties

  /* Additional properties */

  /** Inteter part of log2(prop.maxThreadsPerBlock. this is the maximal number
      such that 1<<log2_thrds_blk <= prop.maxThreadsPerBlock */
  int log2_thrds_blk;
};

extern GlobalPtr<GPUInfo> cuda_dev;

#if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE

#include "mex.h"

//* Check if an error accurred before this line
inline cudaError_t gpuPrevErrChk(const char * file, 
				 int line, 
				 bool abort=true) {
  (void) abort;
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    const char *mex_id = "CUDA:prev_err";
    const char *err_name = cudaGetErrorName(err);
    const char *err_string = cudaGetErrorString(err);

    mexPrintf("%s:%d CUDA error (%s - %d) before this point: %s\n\t reported by gpuPrevErrChk()at  %s:%d\n",
	      file, line, err_name, int(err), err_string, __FILE__, __LINE__ );

    mexErrMsgIdAndTxt(mex_id, "%s:%d Error before this point (%s - %d): %s", file, line,
		      err_name, int(err), err_string);
  }
  return err;
}

/** mex_err_args is a parhethesis enclosed list of arguments for mexErrMsgIdAndTxt
    Note that args can contain the variable err_str, defined as the string contiaing
    __FILE__: __LINE__, CUDA_error_string
*/

inline cudaError_t gpuAssert(cudaError_t code,
		      const char *mex_id,
		      const char *err_msg,
		      const char * file, 
		      int line, 
		      bool abort=true) {
  (void) abort;

  if(code != cudaSuccess) {
    char mex_id_str[1024];

    if(mex_id[0] == '\0') {	// Empty mex_id
      sprintf(mex_id_str, "%s:L_%d", file, line);
      mex_id = mex_id_str;
    }
    const char *err_code = cudaGetErrorName(code);
    const char *err_string = cudaGetErrorString(code);
    mexPrintf("%s:%d mex_id is '%s' (%s - %d): %s\n\t reported by gpuAssert(), %s:%d\n",
	      file, line, mex_id, err_code, int(code), err_string, __FILE__, __LINE__);
    cudaGetLastError();		/* Reset last error */
    mexErrMsgIdAndTxt(mex_id,  "%s:%d %s: Cuda error (%s - %d) %s", file, line,
		      err_msg, cudaGetErrorName(code), int(code), cudaGetErrorString(code)); 
  }
  return code;
}


#else  // #if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE

//* Check if an error occurred before this call
inline cudaError_t gpuPrevErrChk(const char * file, 
				 int line, 
				 bool abort=true) {
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    const char *err_name = cudaGetErrorName(err);
    const char *err_string = cudaGetErrorString(err);

    fprintf(stderr,
	    "%s:%d CUDA error (%s - %d) before this point: %s\n\t reported by gpuPrevErrChk()at  %s:%d\n",
	      file, line, err_name, int(err), err_string, __FILE__, __LINE__ );
    if(abort) exit(err);
  }
  return err;
}

inline cudaError_t gpuAssert(cudaError_t code,
		      const char *mex_id,
		      const char *err_msg,
		      const char * file, 
		      int line, 
		      bool abort=true) {
  (void) mex_id;
  if(code != cudaSuccess ) {
    const char *err_name = cudaGetErrorName(code);
    const char *err_string = cudaGetErrorString(code);

    fprintf(stderr, "%s:%d CUDA error: %s\n\t (%s - %d): %s\n\t reported by gpuAssert() at  %s:%d\n",
	    file, line, err_msg, err_name , int(code), err_string, __FILE__, __LINE__);
    cudaGetLastError();		/* Reset last error */
    if(abort) exit(code);
  }
  return code;
}

#endif	// #if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE

#if defined(NDEBUG) && NDEBUG
#define GPU_PREV_ERR_CHK() (cudaSuccess)
#define GPU_ASSERT(code, mex_id, msg) ((void) (mex_id), (void)(msg), (code))

#else

#define GPU_PREV_ERR_CHK() gpuPrevErrChk(__FILE__, __LINE__)
#define GPU_ASSERT(code, mex_id, msg) gpuAssert((code),(mex_id),(msg),__FILE__, __LINE__)

#endif


/** gpuErrChk is an \c assert like macro for GPU errors, suitable for usage both as 
    Matlab MEX and in stand alone mode. The arguments are:
      ans - a cudaError_t value
      mex_id - a Matlabe error ID (a string consisting of at least two
               alphanumeric tokens ('_' is considered alphanumeric)
               separated by ':'. The C++/CUDA code does not check the
               correctness of the format of mex_id.
     err_msg - An error message (may be empty)
*/

#ifndef __HOST_DEVICE__
#define __HOST_DEVICE__ __host__ __device__
#endif

#ifndef __HOST__
#define __HOST__ __host__
#endif

#else  /* __NVCC__ */

#ifndef __HOST_DEVICE__
#define __HOST_DEVICE__
#endif

#ifndef __HOST__
#define __HOST__
#endif

#define GPU_PREV_ERR_CHK() (1)
#define GPU_ASSERT(code, mex_id, msg) ((void) (mex_id), (void)(msg), (code))

#endif /* __NVCC__ */

#define gpuErrChk(ans, mex_id, err_msg) \
  ((void) (GPU_PREV_ERR_CHK() || GPU_ASSERT(ans, mex_id, err_msg)))


#endif	/* __CudaDevInfo_HDR_ _*/

