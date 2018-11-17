/** \file

*/

#include <stdio.h>
#include <stdlib.h>

#include "CodeDest.h"
#include "CodeDest_tst_do.h"

int main(int argc, char *argv[]) {
  size_t n_vals;
  double *data;

  data = parse_args(argc, argv, n_vals);
  if(data == NULL) {
    help(argv[0]);
    exit(EXIT_FAILURE);
  }

  run_tests(n_vals, data);

  delete [] data;

  return 0;
}

void run_tests_gpu(size_t n_vals, const double *input)
{
  // Do nothing
  (void) n_vals;
  (void) input;
}
