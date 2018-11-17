/*
 * get_randi.c
 *
 * Code generation for function 'get_randi'
 *
 * C source code generated on: Wed Nov 13 14:08:39 2013
 *
 */

/* Include files */
#include "get_rand.h"
#include "get_randi.h"
#include "get_randn.h"
#include "get_randperm.h"
#include "get_randperm1.h"
#include "rand.h"
#include "rng.h"

/* Function Definitions */

/*
 * function [ x, xr, y, yr ] = get_randi( cnt, seed, imax )
 */
void get_randi(int32_T cnt, uint32_T b_seed, uint32_T imax, emxArray_real_T *x,
               struct_T *xr, emxArray_real_T *y, struct_T *yr)
{
  int32_T i0;
  int32_T k;

  /* 'get_randi:2' yr = rng(); */
  rng(&yr->Method, &yr->Seed, yr->State, yr->LegacyRandnState);

  /* 'get_randi:3' rng(seed); */
  b_rng(b_seed);

  /* 'get_randi:4' xr = rng(); */
  rng(&xr->Method, &xr->Seed, xr->State, xr->LegacyRandnState);

  /* 'get_randi:5' x = randi(imax,1,cnt); */
  c_rand(cnt, x);
  i0 = x->size[1];
  for (k = 0; k < i0; k++) {
    x->data[k] = 1.0 + floor(x->data[k] * (real_T)imax);
  }

  /* 'get_randi:6' rng(yr) */
  c_rng(yr->Method, yr->State, yr->LegacyRandnState);

  /* 'get_randi:7' y = randi(imax,1,cnt); */
  c_rand(cnt, y);
  i0 = y->size[1];
  for (k = 0; k < i0; k++) {
    y->data[k] = 1.0 + floor(y->data[k] * (real_T)imax);
  }
}

/* End of code generation (get_randi.c) */
