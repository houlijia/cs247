#ifndef __CodeDest_HDR__
#define __CodeDest_HDR__

/** \file

 */

#include <stddef.h>
#include <math.h>
#include <assert.h>

#include "CudaDevInfo.h"
#include "cumsum.h"

#ifdef MATLAB_MEX_FILE
#include "matrix.h"
#endif

class CodeDest;

#ifdef __NVCC__
template <class T>
__global__ static void d_lengthUInt(size_t n_vec,
				    const T* vec,
				    size_t *len
				    );

template <class T>
__global__ static void d_lengthSInt(size_t n_vec,
				    const T* vec,
				    size_t *len
				    );

template <class T>
__global__ static void d_lengthSInt(size_t n_vec,
				    const T* vec,
				    size_t *len,
				    T ofst	//!< offset to subtract from input vector
				    );

template <class T>
  __global__ static
  void d_encodeUInt(size_t n_vec, //!< length of vector
		    const T *vec, //!< input vector
		    const size_t *ends, /**<  end indices vector of size \c n_vec. */
		    unsigned char *code //!< Output array of size \c ends[n_vec]
		    );

template <class T>
  __global__ static
  void d_encodeSInt(size_t n_vec, //!< length of vector
		    const T *vec, //!< input vector
		    const size_t *ends, /**<  end indices vector of size \c n_vec. */
		    unsigned char *code //!< Output array of size \c ends[n_vec]
		    );

template <class T>
  __global__ static
  void d_encodeSInt(size_t n_vec, //!< length of vector
		    const T *vec, //!< input vector
		    const size_t *ends, /**<  end indices vector of size \c n_vec. */
		    unsigned char *code, //!< Output array of size \c ends[n_vec]
		    T ofst	//!< offset to subtract from input vector
		    );

template <class T>
__HOST__ void
h_lengthUInt(size_t n_vec, //!< length of vector
	     const T *vec, //!< input vector
	     bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	     size_t *ends	/**< output vector of size \c n_vec.
				   returns the end indices */
	     ) {
  if(n_vec == 0) return;

  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_lengthUInt <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, ends);
  cudaDeviceSynchronize();
  h_cumsum(n_vec, ends);
}

template <class T>
__HOST__ void
h_lengthSInt(size_t n_vec, //!< length of vector
	     const T *vec, //!< input vector
	     bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	     size_t *ends	/**< output vector of size \c n_vec.
				   returns the end indices */
	     )  {
  if(n_vec == 0) return;

  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_lengthSInt <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, ends);
  cudaDeviceSynchronize();
  h_cumsum(n_vec, ends);
}

template <class T>
__HOST__ void
h_lengthSInt(size_t n_vec, //!< length of vector
	     const T *vec, //!< input vector
	     bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	     size_t *ends,	/**< output vector of size \c n_vec.
				   returns the end indices */
	     T ofst	//!< offset to subtract from input vector
	     )  {
  if(n_vec == 0) return;

  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_lengthSInt <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, ends, ofst);
  cudaDeviceSynchronize();
  h_cumsum(n_vec, ends);
}

template <class T>
__global__ static void d_integerizeNumber(size_t n_vec,
					  const T* vec,
					  double *mnts,
					  int *expnt
					  );

#endif

class CodeDest {

public:

  //! Encode an unsigned integer array, assuming that the length have already
  //! been computed.
  template <class T>
  static __HOST__ void
  encodeUInt(size_t n_vec, //!< length of vector
	     const T *vec, //!< input vector
	     bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	     const size_t *ends, /**<  end indices vector of size \c n_vec.
				    */
	     unsigned char *code //!< Output array of size \c ends[n_vec]
	     )
  {
    if(n_vec == 0) return;

    if(!on_gpu) {
      size_t k;
      for(k=0; k<n_vec; k++)
        CodeDest::encodeUIntSingle(vec[k], code+ends[k]-1);
    }
    else {
#ifdef __NVCC__
      const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
      int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
      int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

      d_encodeUInt <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, ends, code);
      cudaDeviceSynchronize();
#else
      assert(0);
      (void) n_vec; (void) vec; (void) on_gpu; (void) ends; (void) code;
#endif
    }
  }
 
  //! Encode a signed integer array, assuming that the length have already
  //! been computed.
  template <class T>
  static __HOST__ void
  encodeSInt(size_t n_vec, //!< length of vector
	     const T *vec, //!< input vector
	     bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	     const size_t *ends, /**<  end indices vector of size \c n_vec.
				    */
	     unsigned char *code, //!< Output array of size \c ends[n_vec]
	     T ofst=0		//!< offset to subtract from input vector

	     )
  {
    if(n_vec == 0) return;

    if(!on_gpu) {
      size_t k;

      if(ofst) {
	for(k=0; k<n_vec; k++)
	  CodeDest::encodeSIntSingle(vec[k]-ofst, code+ends[k]-1);
      } else {
	for(k=0; k<n_vec; k++)
	  CodeDest::encodeSIntSingle(vec[k], code+ends[k]-1);
      }
    }
    else {
#ifdef __NVCC__
      const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
      int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
      int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

      if(ofst)
	d_encodeSInt <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, ends, code, ofst);
      else
	d_encodeSInt <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, ends, code);
      cudaDeviceSynchronize();
#else
      assert(0);
      (void) n_vec; (void) vec; (void) on_gpu; (void) ends; (void) code;
#endif
    }
  }

  static __HOST__ size_t codeLen(size_t n_vec, const size_t *ends, bool on_gpu)
  {
    size_t cln;

    if(n_vec == 0)
      cln = 0;
    else if(on_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaMemcpy(&cln, &ends[n_vec-1], sizeof(ends[0]),
			   cudaMemcpyDeviceToHost), "CodeDest:cudaMemcpy", "");
#else
      assert(0);
      cln = 0;
#endif
    }
    else
      cln = ends[n_vec-1];
    return cln;
  }
 
  //! Compute  end addresses of byte array produced unsgined int conversion of
  //! a vector of type \c T
  template <class T>
  static __HOST__ void
  endIndxUInt(size_t n_vec, //!< length of vector
	      const T *vec, //!< input vector
	      bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	      size_t *ends	/**< output vector of size \c n_vec.
				   returns the end indices */
	      )
  {
    if(!on_gpu) {
      size_t k;
      for(k=0; k<n_vec; k++)
	ends[k] = lengthSingleUInt(vec[k]);
      c_cumsum(n_vec, ends, ends);
    }
    else {
#ifdef __NVCC__
      h_lengthUInt(n_vec, vec, on_gpu, ends);
#else
      assert(0);
      (void) n_vec; (void) vec; (void) on_gpu; (void) ends;
#endif
    }
  }

  //! Compute  end addresses of byte array produced sgined int conversion of
  //! a vector of type \c T
  template <class T>
  static __HOST__ void
  endIndxSInt(size_t n_vec, //!< length of vector
	      const T *vec, //!< input vector
	      bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	      size_t *ends	/**< output vector of size \c n_vec.
				   returns the end indices */
	      )
  {
    if(!on_gpu) {
      size_t k;

      for(k=0; k<n_vec; k++)
	ends[k] = lengthSingleSInt(vec[k]);
 
     c_cumsum(n_vec, ends, ends);
    }
    else {
#ifdef __NVCC__
      h_lengthSInt(n_vec, vec, on_gpu, ends);
#else
      assert(0);
      (void) n_vec; (void) vec; (void) on_gpu; (void) ends;
#endif
     }
  }

//! Compute  end addresses of byte array produced sgined int conversion of
  //! a vector of type \c T
  template <class T>
  static __HOST__ void
  endIndxSInt(size_t n_vec, //!< length of vector
	      const T *vec, //!< input vector
	      bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
	      size_t *ends,	/**< output vector of size \c n_vec.
				   returns the end indices */
	      T ofst		/**< Optional offset to subtract from input */
	      )
  {
    if(!on_gpu) {
      size_t k;

      for(k=0; k<n_vec; k++)
	ends[k] = lengthSingleSInt(vec[k]-ofst);
 
      c_cumsum(n_vec, ends, ends);
    }
    else {
#ifdef __NVCC__
      h_lengthSInt(n_vec, vec, on_gpu, ends, ofst);
#else
      assert(0);
      (void) n_vec; (void) vec; (void) on_gpu; (void) ends; (void) ofst;
#endif
     }
  }

  template <class T>
    __HOST_DEVICE__ static size_t lengthSingleUInt(T val) {
    return encodeIntLength(val, 7, 7);
  }

  template <class T>
  __HOST_DEVICE__ static size_t
  lengthSingleSInt(T val) {
    if(val < 0)
      val = -val;
    return encodeIntLength(val, 7, 6);
  }

 template <class T>
  __HOST_DEVICE__ static size_t
   lengthSingleSInt(T val, T ofst) {
    if(val < 0)
      val = -val;
    return encodeIntLength(val-ofst, 7, 6);
  }

   template <class T>
  __HOST_DEVICE__ static void encodeUIntSingle(T val, unsigned char *p_end)
  {
    *p_end-- = val & T(0x7F);
    val >>= 7;
    while(val) {
      *p_end-- = ((unsigned char)0x80) | (unsigned char)(val & T(0x7F));
      val >>= 7;
    }
  }

  template <class T>
  __HOST_DEVICE__ static void encodeSIntSingle(T val, unsigned char *p_end)
  {
    unsigned char sgn = (val >= 0)? 0: 0x40;
    if(sgn)
      val = -val;
   if(val < T(0x40)) {
      *p_end = sgn | ((unsigned char)val);
      return;
    }
    *p_end-- = val & T(0x7F);
    for(val >>= 7;val >= 0x40; val >>= 7)
      *p_end-- = ((unsigned char)0x80) | (unsigned char)(val & T(0x7F));
    *p_end = ((unsigned char)0x80) | sgn | ((unsigned char)val); 
  }

   // "template" here for use with float and double
  template <class T>
  static __HOST__ void
  integerizeNumber(size_t n_vec, //!< length of vector
		   const T *vec, //!< input vector
		   bool on_gpu,	/**< If true the vector is on GPU
				   should be false if not compiled as CUDA) */
		   double *mnts, /**< mantissa output vector of size \c n_vec. */
		   int *expnt /**<exponent output vector of size \c n_vec. */
		   ) {
    if(!on_gpu) {
      for(size_t k=0; k<n_vec; k++)
	integerizeSingleNumber(vec[k], mnts[k], expnt[k]);
    }
    else {
#ifdef __NVCC__
      if(n_vec) {
	const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
	int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;
	d_integerizeNumber <<< n_blks, n_thrds_per_blk >>> (n_vec, vec, mnts, expnt);
	cudaDeviceSynchronize();
      }
#else
      assert(0);
#endif
    }
  }

  __HOST_DEVICE__ static void integerizeSingleNumber(double val,
						     double &mnts,
						     int &expnt
						     ) {
    double f;
    int e;

    f = frexp(val, &e);
    if (f != 0) {
      double f1 = f;
      int e1 = e, m = 3;
      const int e_min  = -1022;

      while(f1>floor(f1) && e1 > e_min) {
	f = f1; e = e1;
	m = (2*m+1) < (e1-e_min)? (2*m+1): (e1-e_min);
	f1 = ldexp(f1, m);
	e1 -= m;
      }
      while(m > 1) {
	int mm = m>>1;
	double ff = ldexp(f,mm);
	int ee = e - mm;
	if(ff > floor(ff)) {
	  f = ff; e = ee;
	  m = m - mm;
	} else {
	  f1 = ff; e1 = ee;
	  m = mm;
	}
      }
      f = floor(f1);
      e = e1;
    }
    mnts = f;
    expnt = e;
  }

  __HOST_DEVICE__ static void integerizeSingleNumber(float val,
						     double &mnts,
						     int &expnt
						     )
  { CodeDest::integerizeSingleNumber(double(val), mnts, expnt); }


private:

  template <class T>
  __HOST_DEVICE__ static void encodeIntGetSign(T &val, unsigned char &sgn) {
    if(val < 0) { val = -val; sgn = 64;}
    else sgn = 0;
  }

  template <class T>
  __HOST_DEVICE__ static size_t encodeIntLength(T val, int shft, int shft1) {
    size_t sz = 1;
    for(val = val >> shft1; val; val = val >> shft)
      sz++;
    return sz;
  }

};				// End of class declaration
    

#ifdef __NVCC__
template <class T>
__global__ static
void d_encodeUInt(size_t n_vec, //!< length of vector
		  const T *vec, //!< input vector
		  const size_t *ends, /**<  end indices vector of size \c n_vec. */
		  unsigned char *code //!< Output array of size \c ends[n_vec]
		  )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec)
    CodeDest::encodeUIntSingle(vec[indx], code+ends[indx]-1);
}

template <class T>
__global__ static
void d_encodeSInt(size_t n_vec, //!< length of vector
		  const T *vec, //!< input vector
		  const size_t *ends, /**<  end indices vector of size \c n_vec. */
		  unsigned char *code //!< Output array of size \c ends[n_vec]
		  )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec)
    CodeDest::encodeSIntSingle(vec[indx], code+ends[indx]-1);
}

template <class T>
  __global__ static
  void d_encodeSInt(size_t n_vec, //!< length of vector
		    const T *vec, //!< input vector
		    const size_t *ends, /**<  end indices vector of size \c n_vec. */
		    unsigned char *code, //!< Output array of size \c ends[n_vec]
		    T ofst	//!< offset to subtract from input vector
		    )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec)
    CodeDest::encodeSIntSingle(vec[indx]-ofst, code+ends[indx]-1);
}

#endif

template <>
__HOST_DEVICE__ inline void CodeDest::encodeUIntSingle<double>(double val,
							       unsigned char *p_end
							       )
{
  if(val != ceil(val))
    val = ceil(val - 0.5);
  double vshft = floor(val/128.);
  *p_end-- = (unsigned char)(val - vshft*128.);
  for(val = vshft; val != 0; val = vshft) {
    vshft = floor(val/128.);
    *p_end-- = ((unsigned char)0x80) | (unsigned char)(val - vshft*128.);
  }
}

template <>
__HOST_DEVICE__ inline void  CodeDest::encodeUIntSingle<float>(float val,
							       unsigned char *p_end
							       )
{
#ifndef __NVCC__
  CodeDest::encodeUIntSingle(double(val), p_end);
#else
  val = rintf(val);
  float vshft = floorf(val/128.f);
  *p_end-- = (unsigned char)(val - vshft*128.f);
  for(val = vshft; val != 0.f; val = vshft) {
    vshft = floor(val/128.f);
    *p_end-- = ((unsigned char)0x80) | (unsigned char)(val - vshft*128.f);
  }
#endif
}

#ifndef MATLAB_MEX_FILE
typedef bool mxLogical;
#endif

template <>
__HOST_DEVICE__ inline void CodeDest::encodeUIntSingle<mxLogical>(mxLogical val,
								  unsigned char *p_end
								  )
{
  *p_end = (unsigned char)(val?1:0);
}

template <>
__HOST_DEVICE__ inline void CodeDest::encodeSIntSingle<double>(double val,
							       unsigned char *p_end
							       )
{
  if(val != ceil(val))
    val = ceil(val - 0.5);
  unsigned char sgn = (val >= 0)? 0: 0x40;
  if(sgn)
    val = -val;
    
  if(val < double(0x40)) {
    *p_end = sgn | ((unsigned char)val);
    return;
  }
  double vshft = floor(val/128.);
  *p_end-- = (unsigned char)(val - vshft*128.);
  for(val = vshft; val >= double(0x40); val = vshft) {
    vshft = floor(val/128.);
    *p_end-- = ((unsigned char)0x80) | (unsigned char)(val - vshft*128.);
  }
  *p_end = ((unsigned char)0x80) | sgn | ((unsigned char)val); 
}


template <>
__HOST_DEVICE__ inline void CodeDest::encodeSIntSingle<float>(float val,
							      unsigned char *p_end
							      )
{
#ifndef __NVCC__
  CodeDest::encodeSIntSingle(double(val), p_end);
#else
  val = rintf(val);
  unsigned char sgn = (val >= 0)? 0: 0x40;
  if(sgn)
    val = -val;
  if(val < float(0x40)) {
    *p_end = sgn | ((unsigned char)val);
    return;
  }
  float vshft = floorf(val/128.f);
  *p_end-- = (unsigned char)(val - vshft*128.f);
  for(val = vshft; val >= float(0x40); val = vshft) {
    vshft = floorf(val/128.f);
    *p_end-- = ((unsigned char)0x80) | (unsigned char)(val - vshft*128.f);
  }
  *p_end = ((unsigned char)0x80) | sgn | ((unsigned char)val);
#endif 
}


template <>
__HOST_DEVICE__ inline size_t CodeDest::encodeIntLength<double>(double val,
								int shft,
								int shft1) {
  size_t sz = 1;
  int expnt;
  if(frexp(val, &expnt) != 0 && expnt > shft1)
    sz += (expnt+shft-shft1-1)/shft;
  return sz;
}

template <>
__HOST_DEVICE__ inline size_t CodeDest::encodeIntLength<float>(float val,
							       int shft,
							       int shft1) {
#ifdef __CUDA_ARCH__
  size_t sz = 1;
  int expnt;
  if(frexpf(val, &expnt) != 0 && expnt > shft1)
    sz += (expnt+shft-shft1-1)/shft;
  return sz;
#else
  return CodeDest::encodeIntLength((double)val, shft, shft1);
#endif
}

#ifdef MATLAB_MEX_FILE
template <>
__HOST_DEVICE__ inline size_t CodeDest::encodeIntLength<mxLogical>(mxLogical val,
								int shft,
								int shft1) {
  (void) val; (void) shft; (void) shft1;
  return 1;
}
#endif

#ifdef __NVCC__
template <class T>
__global__ static void d_lengthUInt(size_t n_vec,
				    const T* vec,
				    size_t *len
				    )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec) {
    len[indx] = CodeDest::lengthSingleUInt(vec[indx]);
  }
}

template <class T>
__global__ static void d_lengthSInt(size_t n_vec,
				    const T* vec,
				    size_t *len
				    )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec)
    len[indx] = CodeDest::lengthSingleSInt(vec[indx]);
}

template <class T>
__global__ static void d_lengthSInt(size_t n_vec,
				    const T* vec,
				    size_t *len,
				    T ofst	//!< offset to subtract from input vector
				    )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec)
    len[indx] = CodeDest::lengthSingleSInt(vec[indx]-ofst);
}

template <class T>
__global__ static void d_integerizeNumber(size_t n_vec,
					  const T* vec,
					  double *mnts,
					  int *expnt
					  ) {
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < n_vec)
    CodeDest::integerizeSingleNumber(vec[indx], mnts[indx], expnt[indx]);
}


#endif



#endif	/*  __CodeDest_HDR__ */
