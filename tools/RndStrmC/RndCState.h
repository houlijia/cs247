#ifndef RndCState_h
#define RndCState_h

#include "RndC_ifc.h"
#include "rtwtypes.h"

typedef struct RndCState {
  RndC_uint32 method;
  RndC_uint32 b_method;
  RndC_uint32 state[2];
  RndC_uint32 b_state[625];
  boolean_T state_not_empty;
  RndC_uint32 c_state;
  RndC_uint32 d_state[2];
} RndCState;

void getRndCState(RndCState *rnd_state);
void setRndCState(const RndCState *rnd_state);

#endif	/* RndCState_h */
