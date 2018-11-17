/*
 * get_rand.c
 *
 * Code generation for function 'get_rand'
 *
 * C source code generated on: Wed Nov 13 14:08:38 2013
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
 * function [ x, xr, y, yr ] = get_rand( cnt, seed )
 */
void get_rand(uint32_T cnt, uint32_T b_seed, emxArray_real_T *x, struct_T *xr,
              emxArray_real_T *y, struct_T *yr)
{
  /* 'get_rand:2' yr = rng(); */
  rng(&yr->Method, &yr->Seed, yr->State, yr->LegacyRandnState);

  /* 'get_rand:3' rng(seed); */
  b_rng(b_seed);

  /* 'get_rand:4' xr = rng(); */
  rng(&xr->Method, &xr->Seed, xr->State, xr->LegacyRandnState);

  /* 'get_rand:5' x = rand(1,cnt); */
  b_rand(cnt, x);

  /* 'get_rand:6' rng(yr) */
  c_rng(yr->Method, yr->State, yr->LegacyRandnState);

  /* 'get_rand:7' y = rand(1,cnt); */
  b_rand(cnt, y);
}

/* End of code generation (get_rand.c) */
