#include <stddef.h>
#include <string.h>

#include <matrix.h>

#include "RndCState_mex.h"
#include "RndCState.h"

const char *RndCState_flds[] = {"method", "b_method", "state", "b_state",
			      "state_not_empty", "c_state", "d-state"};

#define N_FLDS 7

mxArray *RndCState_to_mex(const struct RndCState *rnd_state)
{

  mxArray *mx_rnd_state;
  mxArray *mx_fld_vals[N_FLDS];
  size_t k;

  /* method */
  mx_fld_vals[0] = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
  memcpy(mxGetData(mx_fld_vals[0]), &rnd_state->method, sizeof(rnd_state->method));

  /* b_method */
  mx_fld_vals[1] = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
  memcpy(mxGetData(mx_fld_vals[1]), &rnd_state->b_method, sizeof(rnd_state->b_method));

  /* state */
  mx_fld_vals[2] = mxCreateNumericMatrix(1,2, mxUINT32_CLASS, mxREAL);
  memcpy(mxGetData(mx_fld_vals[2]), &rnd_state->state, sizeof(rnd_state->state));

  /* b_state */
  mx_fld_vals[3] = mxCreateNumericMatrix(1,625, mxUINT32_CLASS, mxREAL);
  memcpy(mxGetData(mx_fld_vals[3]), &rnd_state->b_state, sizeof(rnd_state->b_state));

  /* state_not_empty */
  mx_fld_vals[4] = mxCreateLogicalScalar(rnd_state->state_not_empty);

  /* c_state*/
  mx_fld_vals[5] = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
  memcpy(mxGetData(mx_fld_vals[5]), &rnd_state->c_state, sizeof(rnd_state->c_state));

  /* d_state */
  mx_fld_vals[6] = mxCreateNumericMatrix(1,2, mxUINT32_CLASS, mxREAL);
  memcpy(mxGetData(mx_fld_vals[6]), &rnd_state->d_state, sizeof(rnd_state->d_state));

  mx_rnd_state = mxCreateStructMatrix(1,1, N_FLDS, RndCState_flds);

  for (k=0; k<N_FLDS; k++)
    mxSetFieldByNumber(mx_rnd_state, 0, k, mx_fld_vals[k]);

  return mx_rnd_state;
}

void RndCSTate_from_mex(struct RndCState *rnd_state, const mxArray *mx_rnd_state)
{
  const mxArray *mx_fld_vals[N_FLDS];
  size_t k;

  for(k=0; k<N_FLDS; k++)
    mx_fld_vals[k] = mxGetField(mx_rnd_state, 0, RndCState_flds[k]);

 /* method */
  memcpy(&rnd_state->method, mxGetData(mx_fld_vals[0]), sizeof(rnd_state->method));

  /* b_method */
  memcpy(&rnd_state->b_method, mxGetData(mx_fld_vals[1]), sizeof(rnd_state->b_method));

  /* state */
  memcpy(&rnd_state->state, mxGetData(mx_fld_vals[2]), sizeof(rnd_state->state));

  /* b_state */
  memcpy(&rnd_state->b_state, mxGetData(mx_fld_vals[3]), sizeof(rnd_state->b_state));

  /* state_not_empty */
  rnd_state->state_not_empty = mxIsLogicalScalarTrue(mx_fld_vals[4]);

  /* c_state*/
  memcpy(&rnd_state->c_state, mxGetData(mx_fld_vals[5]), sizeof(rnd_state->c_state));

  /* d_state */
  memcpy(&rnd_state->d_state, mxGetData(mx_fld_vals[6]), sizeof(rnd_state->d_state));
}
