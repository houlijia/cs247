/*
 * rng.c
 *
 * Code generation for function 'rng'
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
#include "rng.h"
#include "rand.h"
#include "get_rand_data.h"

/* Variable Definitions */
static uint32_T seed;
static boolean_T seed_not_empty;

/* Function Declarations */
static uint32_T default_seed(void);
static void eml_rand_mt19937ar_stateful(uint32_T r[625]);

/* Function Definitions */

/*
 *
 */
static uint32_T default_seed(void)
{
  return 0U;
}

/*
 *
 */
static void eml_rand_mt19937ar_stateful(uint32_T r[625])
{
  uint32_T uv0[625];
  if (!state_not_empty) {
    eml_rand_mt19937ar(uv0);
    memcpy(&b_state[0], &uv0[0], 625U * sizeof(uint32_T));
    state_not_empty = TRUE;
  }

  memcpy(&r[0], &b_state[0], 625U * sizeof(uint32_T));
}

/*
 *
 */
void b_rng(uint32_T arg1)
{
  uint32_T r;
  uint32_T uv1[625];
  int32_T b_r;
  int32_T t;
  if (!seed_not_empty) {
    seed = 0U;
    seed_not_empty = TRUE;
  }

  seed = arg1;
  if (method == 7U) {
    if (seed == 0U) {
      seed = 5489U;
    }

    r = seed;
    if (!state_not_empty) {
      eml_rand_mt19937ar(uv1);
      memcpy(&b_state[0], &uv1[0], 625U * sizeof(uint32_T));
      state_not_empty = TRUE;
    }

    b_state[0] = r;
    for (b_r = 0; b_r < 623; b_r++) {
      r = (r ^ r >> 30U) * 1812433253U + (1 + b_r);
      b_state[b_r + 1] = r;
    }

    b_state[624] = 624U;
  } else if (method == 5U) {
    d_state[0] = 362436069U;
    d_state[1] = seed;
    if (d_state[1] == 0U) {
      d_state[1] = 521288629U;
    }
  } else {
    b_r = (int32_T)(seed >> 16U);
    t = (int32_T)(seed & 32768U);
    c_state = (uint32_T)b_r << 16U;
    c_state = seed - c_state;
    c_state -= t;
    c_state <<= 16U;
    c_state += t;
    c_state += b_r;
    if (c_state < 1U) {
      c_state = 1144108930U;
    } else {
      if (c_state > 2147483646U) {
        c_state = 2147483646U;
      }
    }
  }

  b_method = 0U;
}

/*
 *
 */
void c_rng(uint32_T arg1_Method, const uint32_T arg1_State[625], const uint32_T
           arg1_LegacyRandnState[2])
{
  int32_T i;
  uint32_T uv3[625];
  if (!seed_not_empty) {
    seed = 0U;
    seed_not_empty = TRUE;
  }

  if ((arg1_Method & 16384U) != 0U) {
    b_method = 4U;
  } else if ((arg1_Method & 32768U) != 0U) {
    b_method = 5U;
  } else {
    b_method = 0U;
  }

  for (i = 0; i < 2; i++) {
    state[i] = arg1_LegacyRandnState[i];
  }

  i = (int32_T)(arg1_Method & 16383U);
  method = (uint32_T)i;
  if ((uint32_T)i == 7U) {
    if (!state_not_empty) {
      eml_rand_mt19937ar(uv3);
      memcpy(&b_state[0], &uv3[0], 625U * sizeof(uint32_T));
      state_not_empty = TRUE;
    }

    memcpy(&b_state[0], &arg1_State[0], 625U * sizeof(uint32_T));
  } else if ((uint32_T)i == 5U) {
    d_state[0] = arg1_State[0];
    d_state[1] = arg1_State[1];
  } else {
    if ((uint32_T)i == 4U) {
      c_state = arg1_State[0];
    }
  }
}

/*
 *
 */
void rng(uint32_T *settings_Method, uint32_T *settings_Seed, uint32_T
         settings_State[625], uint32_T settings_LegacyRandnState[2])
{
  uint32_T e_state[625];
  int32_T i;
  if (!seed_not_empty) {
    seed = default_seed();
    seed_not_empty = TRUE;
  }

  if (b_method == 4U) {
    *settings_Method = method | 16384U;
  } else if (b_method == 5U) {
    *settings_Method = method | 32768U;
  } else {
    *settings_Method = method;
  }

  if (method == 7U) {
    eml_rand_mt19937ar_stateful(e_state);
  } else {
    memset(&e_state[0], 0, 625U * sizeof(uint32_T));
    if (method == 4U) {
      e_state[0] = c_state;
    } else {
      for (i = 0; i < 2; i++) {
        e_state[i] = d_state[i];
      }
    }
  }

  *settings_Seed = seed;
  memcpy(&settings_State[0], &e_state[0], 625U * sizeof(uint32_T));
  for (i = 0; i < 2; i++) {
    settings_LegacyRandnState[i] = state[i];
  }
}

void seed_not_empty_init(void)
{
  seed_not_empty = FALSE;
}

/* End of code generation (rng.c) */
