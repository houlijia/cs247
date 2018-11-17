/*
MATLAB function:
   function rnd_state = init_RndC_mex(seed)
rnd_state - A RndCState struct
seed - seed value (uint32)
*/

/*
MATLAB function:
    function rnd_state = init_RndC_mex(seed)
 */
#include <mex.h>
#include <matrix.h>

#include "RndCState_mex.h"
#include "common_RndC_mex.h"
#include "RndC_ifc.h"
#include "RndCState.h"

void mexFunction(int nlhs,
		 mxArray *plhs[],
		 int nrhs,
		 const mxArray *prhs[]
		 )
{
  RndC_uint32 seed;
  RndCState rnd_state;

  chk_args_cnt_RndC_mex(nlhs, nrhs, "init_RndC_mex", 1, 1);

  seed = get_uint32_mex(prhs[0], "init_RndC_mex:seed");

  init_RndC(&rnd_state, seed);

  if (nlhs > 0)
    plhs[0] = RndCState_to_mex(&rnd_state);
}
