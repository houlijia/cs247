/** \file

*/

#include <stdio.h>

#include "CudaDevInfo.h"
#include "CodeDest_tst_do.h"

int main(int argc, char *argv[]) {
  size_t n_vals;
  double *data;

  data = parse_args(argc, argv, n_vals);
  if(data == NULL) {
    help(argv[0]);
    exit(EXIT_FAILURE);
  }

  cudaDeviceReset();

  run_tests(n_vals, data);

  delete [] data;

  return 0;
}

void run_tests_gpu(size_t n_vals, const double *input)
{
  CodeDest_tst_rslt gpu_rslt(n_vals, input, true);
  gpu_rslt.print();
}
