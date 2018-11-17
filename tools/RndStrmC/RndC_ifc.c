#include "RndC_ifc.h"
#include "RndCState.h"
#include "get_rand_initialize.h"
#include "rng.h"
#include "get_rand_types.h"
#include "get_rand_emxAPI.h"
#include "rand.h"
#include "randn.h"
#include "randperm.h"

void init_RndC(struct RndCState *rnd_state, RndC_uint32 seed)
{
  get_rand_initialize();
  b_rng(seed);
  getRndCState(rnd_state);
}

/* Get an array of uniformly distributed random variables in [0,1] */
void rand_RndC(struct RndCState *rnd_state,  /* generator state, modified */
	       size_t cnt,	/* Length of output array */
	       double *out	/* output (cnt entries) */
	       )
{
  emxArray_real_T *x;

  x = emxCreateWrapper_real_T(out, 1, (RndC_uint32)cnt);
  setRndCState(rnd_state);
  b_rand((RndC_uint32)cnt, x);
  getRndCState(rnd_state);
  emxDestroyArray_real_T(x);
}

/* Get an array of uniformly distributed random integers in  in [1,imax */
void randi_RndC(struct RndCState *rnd_state,  /* generator state, modified */
                RndC_uint32 imax,	/* Maximum of random variables */
	        size_t cnt,	/* Length of output array */
	        RndC_uint32 *out	/* output (cnt entries) */
	        )
{
  emxArray_real_T *x;
  size_t k;

  x = emxCreate_real_T(1, (RndC_uint32)cnt);
  setRndCState(rnd_state);
  b_rand((RndC_uint32)cnt, x);
  getRndCState(rnd_state);
  for (k=0; k<cnt; k++)
    out[k] = (RndC_uint32)(1.0 + floor(x->data[k] * (double)imax));

  emxDestroyArray_real_T(x);

}

/* Get an array of normally distributed random variables (mean=0, var=1) */
void randn_RndC(struct RndCState *rnd_state,  /* generator state, modified */
		size_t cnt,	/* Length of output array */
		double *out	/* output (cnt entries) */
		)
{
  emxArray_real_T *x;

  x = emxCreateWrapper_real_T(out, 1, (RndC_uint32)cnt);
  setRndCState(rnd_state);
  randn((RndC_uint32)cnt, x);
  getRndCState(rnd_state);
  emxDestroyArray_real_T(x);
}

/* Get first cnt entries of a random permutation of 1,...,imax */
void randperm_RndC(struct RndCState *rnd_state,  /* generator state, modified */
		   size_t imax,
		   size_t cnt,	/* Length of output array */
		   RndC_uint32 *out	/* output (cnt entries) */
		   )
{
  emxArray_real_T *x;
  size_t k;

  x = emxCreate_real_T(1, (RndC_uint32)cnt);
  setRndCState(rnd_state);
  randperm((RndC_uint32)imax, (RndC_uint32)cnt, x);
  getRndCState(rnd_state);
  for (k=0; k<cnt; k++)
    out[k] = (RndC_uint32)(x->data[k]);

  emxDestroyArray_real_T(x);
}

/* Get a random permutation of 1,...,imax */
void randperm1_RndC(struct RndCState *rnd_state,  /* generator state, modified */
		   size_t imax,
		   RndC_uint32 *out	/* output (imax entries) */
		   )
{
  emxArray_real_T *x;
  size_t k;

  x = emxCreate_real_T(1, (RndC_uint32)imax);
  setRndCState(rnd_state);
  b_randperm((RndC_uint32)imax, x);
  getRndCState(rnd_state);
  for (k=0; k<imax; k++)
    out[k] = (RndC_uint32)(x->data[k]);

  emxDestroyArray_real_T(x);
}
