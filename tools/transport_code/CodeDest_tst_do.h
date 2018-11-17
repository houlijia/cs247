#ifndef __CodeDest_tst_do_HDR__
#define  __CodeDest_tst_do_HDR__

#include <string.h>
#include <assert.h>

#include "CudaDevInfo.h"
#include "print_vec.h"

#include "cc_vec_op_binary.h"
#include "CodeDest.h"

/** \file 
    Functions used by both cc_CodeDest_tst.cc and cuda_CodeDestTest.cu
*/

extern bool default_use_gpu;

void h_compCodeSize(size_t size,
		    bool use_gpu,
		    size_t *code_size,
		    const size_t *ends
		    );

#ifdef __NVCC__
  __global__ static void d_comp_err(size_t sz,
				    const double *inp,
				    const double *mn,
				    const int *ex,
				    double *er
				    ) {
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k < sz)
    er[k] =  ldexp(mn[k], ex[k]) - inp[k];
  }

#endif

template <class T> 
struct CodeDestTestOutput {
  bool &use_gpu;
  size_t &size;
  T* input;
  size_t code_len;
  unsigned char *code;

  CodeDestTestOutput(size_t &sz,
		     const double *inp, //!< reference input
		     //! use_gpu. Ignored if no CUDA
		     bool &use_gp = default_use_gpu
                     ):
    use_gpu(use_gp), size(sz), code_len(0), code(NULL)
  {
      
    if(use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaMalloc(&input, sz*sizeof(T)),
		"CodeDestTestOutput:malloc", "cudaMalloc failed");
#endif
    } else {
      input = new T[size];
      assert(input != NULL);
    }
  }

  ~CodeDestTestOutput() {
    if(use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaFree(code), 
		"CodeDestTestOutput:malloc", "cudaFree failed");
      gpuErrChk(cudaFree(input), 
		"CodeDestTestOutput:malloc", "cudaFree failed");
#endif
    } else {
      delete [] code;
      delete [] input;
    }
  }

  virtual void compute() = 0;

  virtual void print() = 0;

protected:
  virtual void allocCode() = 0;
};    

//! Derived class for integers case
template <class T> 
struct CodeDestTestOutputI : public CodeDestTestOutput<T> {
  size_t *code_size;
  size_t *ends;

  CodeDestTestOutputI(size_t &sz,
		      const double *inp, //!< reference input
		      //! use_gpu. Ignored if no CUDA
		      bool &use_gp = default_use_gpu
                     ):
    CodeDestTestOutput<T>(sz, inp, use_gp)
  {
    if(this->use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaMalloc(&code_size, sz*sizeof(*code_size)),
		"CodeDestTestOutputI:alloc", "cudaMalloc failed");
      gpuErrChk(cudaMalloc(&ends, sz*sizeof(*ends)),
		"CodeDestTestOutputI:alloc", "cudaMalloc failed");
#endif
    } else {
      code_size = new size_t[sz];
      ends = new size_t[sz];
      assert(code_size != NULL && ends != NULL);
    }
  }

  ~CodeDestTestOutputI() {
    if(this->use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaFree(ends), "CodeDestTestOutputI:alloc", "cudaFree failed");
      gpuErrChk(cudaFree(code_size), "CodeDestTestOutputI:alloc", "cudaMalloc failed");
#endif
    } else {
      delete [] ends;
      delete [] code_size;
    }
  }

  virtual void print() {
    printf ("Lengths:\n");
    print_vec(this->size, this->code_size, 10, " %7lu", this->use_gpu);
    printf("Ends:\n");
    print_vec(this->size, this->ends, 10, " %7lu", this->use_gpu);
    printf("Code (length=%lu):\n", (unsigned long)this->code_len);
    print_vec(this->code_len, this->code, 16, " %2X", this->use_gpu);
   }

protected:
  virtual void compCodeSize() {
    if(!this->size)
      return;

    if(!this->use_gpu) {
      memcpy(this->code_size, this->ends, this->size*sizeof(this->ends[0]));
      c_sub_asgn(this->size-1, this->code_size+1, this->ends);
    }
#ifdef __NVCC__
    else {
      h_compCodeSize(this->size, this->use_gpu, this->code_size, this->ends);
    }
#endif
  }

  virtual void allocCode() {
    this->code_len = CodeDest::codeLen(this->size, this->ends, this->use_gpu);
    if(this->use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaMalloc(&this->code, this->code_len),
		"compCodeSize:alloc", "cudaMalloc failed");
#else
      assert(0);
#endif
    }
    else {
      this->code = new unsigned char[this->code_len];
      assert(this->code != NULL);
    }
  }

};

//! Derived class for unsigned case
template <class T> 
struct CodeDestTestOutputU : public CodeDestTestOutputI<T> {

  CodeDestTestOutputU(size_t &sz,
		      const double *inp, //!< reference input
		      //! use_gpu. Ignored if no CUDA
		      bool &use_gp = default_use_gpu
		      ) :
    CodeDestTestOutputI<T>(sz, inp, use_gp) {

    if(this->use_gpu) {
#ifdef __NVCC__
      T *tmp_inp = new T[this->size];
      this->copyFromDouble(inp, tmp_inp);
      gpuErrChk(cudaMemcpy(this->input, tmp_inp, sz*sizeof(T), cudaMemcpyHostToDevice),
		"CodeDestTestOutputU:memcpy", "cudaMemcpy failed");
      delete [] tmp_inp;
#endif
    } else {
      this->copyFromDouble(inp, this->input);
    }
  }

  ~CodeDestTestOutputU() {}

  void compute() {
    // Compute length of each codeword
    CodeDest::endIndxUInt(this->size, this->input, this->use_gpu, this->ends);
    this->compCodeSize();
    
    // Allocate code
    this->allocCode();
       
    // Compute code
    CodeDest::encodeUInt(this->size, this->input, this->use_gpu, this->ends,
			 this->code);
     
  }

private:
  virtual void copyFromDouble(const double *inp, T *vec) {
    size_t k;

    for(k=0; k < this->size; k++)
      vec[k] = T(fabs(inp[k]));
   }

};    

template <class T>
struct CodeDestTestOutputS : public CodeDestTestOutputI<T> {
  CodeDestTestOutputS(size_t &sz,
		      const double *inp, //!< reference input
		      //! use_gpu. Ignored if no CUDA
		      bool &use_gp = default_use_gpu
		      ): CodeDestTestOutputI<T>(sz,inp,use_gp)
  {
    if(this->use_gpu) {
#ifdef __NVCC__
      T *tmp_inp = new T[this->size];
      this->copyFromDouble(inp, tmp_inp);
      gpuErrChk(cudaMemcpy(this->input, tmp_inp, sz*sizeof(T), cudaMemcpyHostToDevice),
		"CodeDestTestOutputS:memcpy", "cudaMemcpy failed");
      delete [] tmp_inp;
#endif
    } else {
      this->copyFromDouble(inp, this->input);
    }
  }

    void compute() {
    // Compute length of each codeword
    CodeDest::endIndxSInt(this->size, this->input, this->use_gpu, this->ends);
    this->compCodeSize();
  
    // Allocate code
    this->allocCode();
      
    // Compute code
    CodeDest::encodeSInt(this->size, this->input, this->use_gpu, this->ends,
			 this->code);
 }

private:
  virtual void copyFromDouble(const double *inp, T *vec) {
    size_t k;

    for(k=0; k < this->size; k++)
      vec[k] = T(inp[k]);
  }
};

template <class T>
struct CodeDestTestOutputF : public CodeDestTestOutput<T> {
  double *mnts;
  int *expnt;
  size_t *mnts_ends;
  size_t *expnt_ends;
  size_t mnts_code_len;
  double *err;

  CodeDestTestOutputF(size_t &sz,
		      const double *inp, //!< reference input
		      //! use_gpu. Ignored if no CUDA
		      bool &use_gp = default_use_gpu
                     ):
    CodeDestTestOutput<T>(sz, inp, use_gp)
  {
    if(this->use_gpu) {
#ifdef __NVCC__
       T *tmp_inp = new T[this->size];
      this->copyFromDouble(inp, tmp_inp);
      gpuErrChk(cudaMemcpy(this->input, tmp_inp, sz*sizeof(T), cudaMemcpyHostToDevice),
		"CodeDestTestOutputF:memcpy", "cudaMemcpy failed");
      delete [] tmp_inp;

      gpuErrChk(cudaMalloc(&mnts, sz*sizeof(*mnts)),
		"CodeDestTestOutputF:alloc", "cudaMalloc failed");
      gpuErrChk(cudaMalloc(&expnt, sz*sizeof(*expnt)),
		"CodeDestTestOutputF:alloc", "cudaMalloc failed");
      gpuErrChk(cudaMalloc(&mnts_ends, sz*sizeof(*mnts_ends)),
		"CodeDestTestOutputF:alloc", "cudaMalloc failed");
      gpuErrChk(cudaMalloc(&expnt_ends, sz*sizeof(*expnt_ends)),
		"CodeDestTestOutputF:alloc", "cudaMalloc failed");
      gpuErrChk(cudaMalloc(&err, sz*sizeof(*err)),
		"CodeDestTestOutputF:alloc", "cudaMalloc failed");
#endif
    } else {
      this->copyFromDouble(inp, this->input);

      mnts = new double[sz];
      expnt = new int[sz];
      mnts_ends = new size_t[sz];
      expnt_ends = new size_t[sz];
      err = new double[sz];
      assert(mnts != NULL && expnt != NULL &&
	     mnts_ends != NULL && expnt_ends != NULL && err != NULL);
    }
  }

  ~CodeDestTestOutputF() {
    if(this->use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaFree(err), 
		"deleteCodeDestTestOutputF:alloc", "cudaFree failed");
      gpuErrChk(cudaFree(expnt_ends),
		"deleteCodeDestTestOutputF:alloc", "cudaFree failed");
      gpuErrChk(cudaFree(mnts_ends),
		"deleteCodeDestTestOutputF:alloc", "cudaFree failed");
      gpuErrChk(cudaFree(expnt),
		"deleteCodeDestTestOutputF:alloc", "cudaFree failed");
      gpuErrChk(cudaFree(mnts),
		"deleteCodeDestTestOutputF:alloc", "cudaFree failed");
#endif
    } else {
      delete [] err;
      delete [] expnt_ends;
      delete [] mnts_ends;
      delete [] expnt;
      delete [] mnts;
    }
  }

  void compute() {
    // compute mantissa and exponent
    CodeDest::integerizeNumber(this->size, this->input, this->use_gpu,
			       this->mnts, this->expnt);

    if(this->use_gpu) {
#ifdef __NVCC__
      if(this->size) {
	const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	int n_blks = (this->size + max_thr_blk -1)/max_thr_blk; 
	int n_thrds_per_blk = (this->size < max_thr_blk)? this->size: max_thr_blk;

	d_comp_err <<< n_blks, n_thrds_per_blk >>> (this->size,
						    this->input,
						    this->mnts,
						    this->expnt,
						    this->err);
	cudaDeviceSynchronize();
       }
#endif
    }
    else {
      for(size_t k=0; k<this->size; k++)
	this->err[k] = ldexp(this->mnts[k], this->expnt[k]) - this->input[k];
    }

    // compute length of each codeword
    CodeDest::endIndxSInt(this->size, this->mnts, this->use_gpu, this->mnts_ends);
    CodeDest::endIndxSInt(this->size, this->expnt, this->use_gpu, this->expnt_ends);

    // Allocate code
    this->allocCode();
      
    // Compute code
    CodeDest::encodeSInt(this->size, this->mnts, this->use_gpu, this->mnts_ends,
			 this->code);
    CodeDest::encodeSInt(this->size, this->expnt, this->use_gpu, this->expnt_ends,
			 this->code + this->mnts_code_len);
  }


  virtual void allocCode() {
    this->mnts_code_len =
      CodeDest::codeLen(this->size, this->mnts_ends, this->use_gpu);
    this->code_len = this->mnts_code_len + 
      CodeDest::codeLen(this->size, this->expnt_ends, this->use_gpu);

    if(this->use_gpu) {
#ifdef __NVCC__
      gpuErrChk(cudaMalloc(&this->code, this->code_len),
		"allocCode:alloc", "cudaMalloc failed");
#else
      assert(0);
#endif
    }
    else {
      this->code = new unsigned char[this->code_len];
      assert(this->code != NULL);
    }
  }

  virtual void print() {
    printf("mantissas:\n");
    print_vec(this->size, this->mnts, 5, " %11.6g", this->use_gpu);
    printf("exponents:\n");
    print_vec(this->size, this->expnt, 5, " %11d", this->use_gpu);
    printf("error:\n");
    print_vec(this->size, this->err, 5, " %11.6g", this->use_gpu);
    printf("Code (length=%lu):\n", (unsigned long)this->code_len);
    print_vec(this->code_len, this->code, 16, " %2X", this->use_gpu);
  }


private:
  virtual void copyFromDouble(const double *inp, T *vec) {
    size_t k;

    for(k=0; k < this->size; k++)
      vec[k] = T(inp[k]);
  }
};

struct CodeDest_tst_rslt {
  size_t size;
  bool use_gpu;

  CodeDestTestOutputU<double> ud;
  CodeDestTestOutputU<float>  uf;
  CodeDestTestOutputU<unsigned long> ul;
  CodeDestTestOutputS<double> sd;
  CodeDestTestOutputS<float>  sf;
  CodeDestTestOutputS<long> sl;
  CodeDestTestOutputF<double> fd;
  CodeDestTestOutputF<double> ff;

  CodeDest_tst_rslt(size_t sz,
		     const double *inp, //!< reference input
		     //! use_gpu. Ignored if no CUDA
		     bool use_gp = default_use_gpu 
		     ) :
    size(sz), use_gpu(use_gp),
    ud(size, inp, use_gpu), uf(size, inp, use_gpu), ul(size, inp, use_gpu), 
    sd(size, inp, use_gpu), sf(size, inp, use_gpu), sl(size, inp, use_gpu),
    fd(size, inp, use_gpu), ff(size, inp, use_gpu)
  {
    ud.compute();
    uf.compute();
    ul.compute();
    sd.compute();
    sf.compute();
    sl.compute();
    fd.compute();
    ff.compute();
  }

  void print() {
    printf("***** Processing on %s ******\n", this->use_gpu? "GPU": "CPU");

    printf("*** Processing unsigned integers ***\n");
    printf("** unsigned double: **\n"); ud.print();
    printf("** unsigned single: **\n"); uf.print();
    printf("** unsigned long: **\n"); ul.print();

    printf("*** Processing signed integers ***\n");
    printf("** signed double: **\n"); sd.print();
    printf("** signed single: **\n"); sf.print();
    printf("** signed long: **\n"); sl.print();

    printf("*** Processing floating point ***\n");
    printf("** double float: **\n"); fd.print();
    printf("** signed float: **\n"); ff.print();
  }
  
};

/** print a help message */
void help(const char *prog_name);

/** Parses the input arguments
    \return a double array of size n_vals, which consists of numbers to process.
*/
double * parse_args(int argc,	//!< No. of arguments. can be 2 to 5.
		    /**
		       Expected values in argv:
		       argv[0]: Program name
		       argv[1]: val0 (double) Initial value
		       argv[2]: n_vals (size_t) number of values [1]
		       argv[3]: mlt (double - mutiplier [1]
		       argv[4]: add (double) - adder [0]		   
		    */
		    char *argv[],
		    size_t &n_vals	//!< [output] Number of elements returned array
		    );

void run_tests(size_t n_vals, const double *input);

#if 0
#ifdef __NVCC__
template <class T>
void __host__ h_print_vec(size_t sz,	//!< number of elements in vector
			  const T vec[],	//!< The vector
			  size_t n_row,	//!< Number of elements to print in a row
			  const char *fmt	//!< Format for a single element
			  //! (including leading spaces)
			  );

#endif


template <class T>  __HOST_DEVICE__
void print_vec(size_t sz,	//!< number of elements in vector
	       const T vec[],	//!< The vector
	       size_t n_row,	//!< Number of elements to print in a row
	       const char *fmt,	//!< Format for a single element (including
				//! leading spaces
	       bool on_gpu = false
	       )
{
  size_t j,k;
  const T *vc = vec;

#ifdef __CUDA_ARCH__
  printf("%s:%d GPU\n", __FILE__, __LINE__);
#else
  printf("%s:%d CPU. on_gpu=%d\n", __FILE__, __LINE__, int(on_gpu));
#endif

#if defined(__NVCC__) && !defined(__CUDA_ARCH__)
  T *vcp;
  if(on_gpu) {
    h_print_vec(sz, vec, n_row, fmt);

    vcp = new T[sz];
    assert(vcp != NULL);
    printf("%s:%d vcp=0x%lX vec=0x%lX, sz=%lu, sizeof=%lu\n", __FILE__, __LINE__,
	   (unsigned long) vcp, (unsigned long) vec, sz, sizeof(T));
    gpuErrChk(cudaMemcpy(vcp, vec, sz*sizeof(T), cudaMemcpyDeviceToHost),
	      "print_vec:memcpy", "cudaMemcpy failed");
    vc = vcp;
  }
#else
  (void) on_gpu;
#endif
  
  printf("%s:%d\n", __FILE__, __LINE__);
  for(j=0; j<sz; j+=n_row) {
    printf("%5lu: ", (unsigned long)j);

    for(k=j; k<j+n_row && k<sz; k++)
      printf(fmt, vc[k]);
    printf("\n");
  }
  if(k != j)
    printf("\n");

#if defined(__NVCC__) && !defined(__CUDA_ARCH__)
  if(on_gpu) {
    printf("%s:%d\n", __FILE__, __LINE__);
    delete vcp;
  }
#endif
}
#endif

void run_tests_gpu(size_t n_vals, const double *input);

#endif	/*  __CodeDest_tst_do_HDR__ */
