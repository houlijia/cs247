/*
 * rand.c
 *
 * Code generation for function 'rand'
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
#include "get_rand_emxutil.h"
#include "get_rand_data.h"

/* Function Declarations */
static real_T b_eml_rand_mt19937ar_stateful(void);

/* Function Definitions */

/*
 *
 */
static real_T b_eml_rand_mt19937ar_stateful(void)
{
  uint32_T uv6[625];
  if (!state_not_empty) {
    eml_rand_mt19937ar(uv6);
    memcpy(&b_state[0], &uv6[0], 625U * sizeof(uint32_T));
    state_not_empty = TRUE;
  }

  return genrandu(b_state);
}

/*
 *
 */
void b_rand(uint32_T varargin_2, emxArray_real_T *r)
{
  int32_T k;
  int32_T hi;
  uint32_T test1;
  uint32_T test2;
  uint32_T uv2[625];
  real_T d0;
  if (method == 4U) {
    k = r->size[0] * r->size[1];
    r->size[0] = 1;
    r->size[1] = (int32_T)varargin_2;
    emxEnsureCapacity((emxArray__common *)r, k, (int32_T)sizeof(real_T));
    for (k = 0; k < (int32_T)varargin_2; k++) {
      hi = (int32_T)(c_state / 127773U);
      test1 = 16807U * (c_state - hi * 127773U);
      test2 = 2836U * hi;
      if (test1 < test2) {
        c_state = (test1 - test2) + 2147483647U;
      } else {
        c_state = test1 - test2;
      }

      r->data[k] = (real_T)c_state * 4.6566128752457969E-10;
    }
  } else if (method == 5U) {
    k = r->size[0] * r->size[1];
    r->size[0] = 1;
    r->size[1] = (int32_T)varargin_2;
    emxEnsureCapacity((emxArray__common *)r, k, (int32_T)sizeof(real_T));
    for (k = 0; k < (int32_T)varargin_2; k++) {
      test1 = 69069U * d_state[0] + 1234567U;
      test2 = d_state[1] ^ d_state[1] << 13;
      test2 ^= test2 >> 17;
      test2 ^= test2 << 5;
      d_state[0] = test1;
      d_state[1] = test2;
      r->data[k] = (real_T)(test1 + test2) * 2.328306436538696E-10;
    }
  } else {
    if (!state_not_empty) {
      eml_rand_mt19937ar(uv2);
      memcpy(&b_state[0], &uv2[0], 625U * sizeof(uint32_T));
      state_not_empty = TRUE;
    }

    k = r->size[0] * r->size[1];
    r->size[0] = 1;
    r->size[1] = (int32_T)varargin_2;
    emxEnsureCapacity((emxArray__common *)r, k, (int32_T)sizeof(real_T));
    for (k = 0; k < (int32_T)varargin_2; k++) {
      d0 = genrandu(b_state);
      r->data[k] = d0;
    }
  }
}

/*
 *
 */
void c_rand(int32_T varargin_2, emxArray_real_T *r)
{
  int32_T k;
  int32_T hi;
  uint32_T test1;
  uint32_T test2;
  uint32_T uv4[625];
  real_T d1;
  if (method == 4U) {
    k = r->size[0] * r->size[1];
    r->size[0] = 1;
    r->size[1] = varargin_2;
    emxEnsureCapacity((emxArray__common *)r, k, (int32_T)sizeof(real_T));
    for (k = 0; k < varargin_2; k++) {
      hi = (int32_T)(c_state / 127773U);
      test1 = 16807U * (c_state - hi * 127773U);
      test2 = 2836U * hi;
      if (test1 < test2) {
        c_state = (test1 - test2) + 2147483647U;
      } else {
        c_state = test1 - test2;
      }

      r->data[k] = (real_T)c_state * 4.6566128752457969E-10;
    }
  } else if (method == 5U) {
    k = r->size[0] * r->size[1];
    r->size[0] = 1;
    r->size[1] = varargin_2;
    emxEnsureCapacity((emxArray__common *)r, k, (int32_T)sizeof(real_T));
    for (k = 0; k < varargin_2; k++) {
      test1 = 69069U * d_state[0] + 1234567U;
      test2 = d_state[1] ^ d_state[1] << 13;
      test2 ^= test2 >> 17;
      test2 ^= test2 << 5;
      d_state[0] = test1;
      d_state[1] = test2;
      r->data[k] = (real_T)(test1 + test2) * 2.328306436538696E-10;
    }
  } else {
    if (!state_not_empty) {
      eml_rand_mt19937ar(uv4);
      memcpy(&b_state[0], &uv4[0], 625U * sizeof(uint32_T));
      state_not_empty = TRUE;
    }

    k = r->size[0] * r->size[1];
    r->size[0] = 1;
    r->size[1] = varargin_2;
    emxEnsureCapacity((emxArray__common *)r, k, (int32_T)sizeof(real_T));
    for (k = 0; k < varargin_2; k++) {
      d1 = genrandu(b_state);
      r->data[k] = d1;
    }
  }
}

/*
 *
 */
real_T d_rand(void)
{
  real_T r;
  int32_T hi;
  uint32_T test1;
  uint32_T test2;
  if (method == 4U) {
    hi = (int32_T)(c_state / 127773U);
    test1 = 16807U * (c_state - hi * 127773U);
    test2 = 2836U * hi;
    if (test1 < test2) {
      c_state = (test1 - test2) + 2147483647U;
    } else {
      c_state = test1 - test2;
    }

    r = (real_T)c_state * 4.6566128752457969E-10;
  } else if (method == 5U) {
    test1 = 69069U * d_state[0] + 1234567U;
    test2 = d_state[1] ^ d_state[1] << 13;
    test2 ^= test2 >> 17;
    test2 ^= test2 << 5;
    d_state[0] = test1;
    d_state[1] = test2;
    r = (real_T)(test1 + test2) * 2.328306436538696E-10;
  } else {
    r = b_eml_rand_mt19937ar_stateful();
  }

  return r;
}

/*
 *
 */
void eml_rand_mt19937ar(uint32_T e_state[625])
{
  uint32_T r;
  int32_T mti;
  memset(&e_state[0], 0, 625U * sizeof(uint32_T));
  r = 5489U;
  e_state[0] = 5489U;
  for (mti = 0; mti < 623; mti++) {
    r = (r ^ r >> 30U) * 1812433253U + (1 + mti);
    e_state[mti + 1] = r;
  }

  e_state[624] = 624U;
}

/*
 *
 */
void genrand_uint32_vector(uint32_T mt[625], uint32_T u[2])
{
  int32_T i;
  uint32_T mti;
  int32_T kk;
  uint32_T y;
  uint32_T b_y;
  uint32_T c_y;
  uint32_T d_y;
  for (i = 0; i < 2; i++) {
    u[i] = 0U;
  }

  for (i = 0; i < 2; i++) {
    mti = mt[624] + 1U;
    if (mti >= 625U) {
      for (kk = 0; kk < 227; kk++) {
        y = (mt[kk] & 2147483648U) | (mt[1 + kk] & 2147483647U);
        if ((int32_T)(y & 1U) == 0) {
          b_y = y >> 1U;
        } else {
          b_y = y >> 1U ^ 2567483615U;
        }

        mt[kk] = mt[397 + kk] ^ b_y;
      }

      for (kk = 0; kk < 396; kk++) {
        y = (mt[kk + 227] & 2147483648U) | (mt[228 + kk] & 2147483647U);
        if ((int32_T)(y & 1U) == 0) {
          c_y = y >> 1U;
        } else {
          c_y = y >> 1U ^ 2567483615U;
        }

        mt[kk + 227] = mt[kk] ^ c_y;
      }

      y = (mt[623] & 2147483648U) | (mt[0] & 2147483647U);
      if ((int32_T)(y & 1U) == 0) {
        d_y = y >> 1U;
      } else {
        d_y = y >> 1U ^ 2567483615U;
      }

      mt[623] = mt[396] ^ d_y;
      mti = 1U;
    }

    y = mt[(int32_T)mti - 1];
    mt[624] = mti;
    y ^= y >> 11U;
    y ^= y << 7U & 2636928640U;
    y ^= y << 15U & 4022730752U;
    y ^= y >> 18U;
    u[i] = y;
  }
}

/*
 *
 */
real_T genrandu(uint32_T mt[625])
{
  real_T r;
  int32_T exitg1;
  uint32_T u[2];
  boolean_T isvalid;
  int32_T k;
  boolean_T exitg2;
  uint32_T b_r;

  /* ========================= COPYRIGHT NOTICE ============================ */
  /*  This is a uniform (0,1) pseudorandom number generator based on:        */
  /*                                                                         */
  /*  A C-program for MT19937, with initialization improved 2002/1/26.       */
  /*  Coded by Takuji Nishimura and Makoto Matsumoto.                        */
  /*                                                                         */
  /*  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,      */
  /*  All rights reserved.                                                   */
  /*                                                                         */
  /*  Redistribution and use in source and binary forms, with or without     */
  /*  modification, are permitted provided that the following conditions     */
  /*  are met:                                                               */
  /*                                                                         */
  /*    1. Redistributions of source code must retain the above copyright    */
  /*       notice, this list of conditions and the following disclaimer.     */
  /*                                                                         */
  /*    2. Redistributions in binary form must reproduce the above copyright */
  /*       notice, this list of conditions and the following disclaimer      */
  /*       in the documentation and/or other materials provided with the     */
  /*       distribution.                                                     */
  /*                                                                         */
  /*    3. The names of its contributors may not be used to endorse or       */
  /*       promote products derived from this software without specific      */
  /*       prior written permission.                                         */
  /*                                                                         */
  /*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    */
  /*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      */
  /*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  */
  /*  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT  */
  /*  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,  */
  /*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT       */
  /*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  */
  /*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  */
  /*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT    */
  /*  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE */
  /*  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */
  /*                                                                         */
  /* =============================   END   ================================= */
  do {
    exitg1 = 0;
    genrand_uint32_vector(mt, u);
    r = 1.1102230246251565E-16 * ((real_T)(u[0] >> 5U) * 6.7108864E+7 + (real_T)
                                  (u[1] >> 6U));
    if (r == 0.0) {
      if ((mt[624] >= 1U) && (mt[624] < 625U)) {
        isvalid = TRUE;
      } else {
        isvalid = FALSE;
      }

      if (isvalid) {
        isvalid = FALSE;
        k = 1;
        exitg2 = FALSE;
        while ((exitg2 == FALSE) && (k < 625)) {
          if (mt[k - 1] == 0U) {
            k++;
          } else {
            isvalid = TRUE;
            exitg2 = TRUE;
          }
        }
      }

      if (!isvalid) {
        b_r = 5489U;
        mt[0] = 5489U;
        for (k = 0; k < 623; k++) {
          b_r = (b_r ^ b_r >> 30U) * 1812433253U + (1 + k);
          mt[k + 1] = b_r;
        }

        mt[624] = 624U;
      }
    } else {
      exitg1 = 1;
    }
  } while (exitg1 == 0);

  return r;
}

/* End of code generation (rand.c) */
