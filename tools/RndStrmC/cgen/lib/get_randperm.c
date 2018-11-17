/*
 * get_randperm.c
 *
 * Code generation for function 'get_randperm'
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
#include "randperm.h"
#include "rng.h"

/* Function Definitions */

/*
 * function [ x, xr, y, yr ] = get_randperm( cnt, imax, seed )
 */
void get_randperm(uint32_T cnt, uint32_T imax, uint32_T b_seed, emxArray_real_T *
                  x, struct_T *xr, emxArray_real_T *y, struct_T *yr)
{
  /* 'get_randperm:2' yr = rng(); */
  rng(&yr->Method, &yr->Seed, yr->State, yr->LegacyRandnState);

  /* 'get_randperm:3' rng(seed); */
  b_rng(b_seed);

  /* 'get_randperm:4' xr = rng(); */
  rng(&xr->Method, &xr->Seed, xr->State, xr->LegacyRandnState);

  /* 'get_randperm:5' x = randperm(imax, cnt); */
  randperm(imax, cnt, x);

  /* 'get_randperm:6' rng(yr) */
  c_rng(yr->Method, yr->State, yr->LegacyRandnState);

  /* 'get_randperm:7' y = randperm(imax, cnt); */
  randperm(imax, cnt, y);
}

/* End of code generation (get_randperm.c) */
