#include "CodeDest_tst_do.cf"

#include "CudaDevInfo.h"
#include "cuda_vec_op_binary.h"

void h_compCodeSize(size_t size,
		    bool use_gpu,
		    size_t *code_size,
		    const size_t *ends
		    )
{
  gpuErrChk(cudaMemcpy(code_size, ends, size*sizeof(ends[0]), cudaMemcpyDeviceToDevice),
	    "h_compCodeSize:alloc", ""	);
  h_sub_asgn(size-1, code_size+1, ends);
}

