/*
 * get_randn.c
 *
 * Code generation for function 'get_randn'
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
#include "randn.h"
#include "rng.h"

/* Function Definitions */

/*
 * function [ x, xr, y, yr ] = get_randn( cnt, seed )
 */
void get_randn(uint32_T cnt, uint32_T b_seed, emxArray_real_T *x, struct_T *xr,
               emxArray_real_T *y, struct_T *yr)
{
  /* 'get_randn:2' yr = rng(); */
  rng(&yr->Method, &yr->Seed, yr->State, yr->LegacyRandnState);

  /* 'get_randn:3' rng(seed); */
  b_rng(b_seed);

  /* 'get_randn:4' xr = rng(); */
  rng(&xr->Method, &xr->Seed, xr->State, xr->LegacyRandnState);

  /* 'get_randn:5' x = randn(1,cnt); */
  randn(cnt, x);

  /* 'get_randn:6' rng(yr) */
  c_rng(yr->Method, yr->State, yr->LegacyRandnState);

  /* 'get_randn:7' y = randn(1,cnt); */
  randn(cnt, y);
}

/* End of code generation (get_randn.c) */
