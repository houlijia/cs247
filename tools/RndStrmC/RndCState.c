#include "get_rand_data.h"
#include "RndCState.h"


void getRndCState(RndCState *rnd_state) {
  rnd_state->method = method;
  rnd_state->b_method = b_method;
  memcpy(rnd_state->state, state, sizeof(state));
  memcpy(rnd_state->b_state, b_state, sizeof(b_state));
  rnd_state->state_not_empty = state_not_empty;
  rnd_state->c_state = c_state;
  memcpy(rnd_state->d_state, d_state, sizeof(d_state));
}

void setRndCState(const RndCState *rnd_state) {
  method = rnd_state->method;
  b_method = rnd_state->b_method;
  memcpy(state, rnd_state->state, sizeof(state));
  memcpy(b_state, rnd_state->b_state, sizeof(b_state));
  state_not_empty = rnd_state->state_not_empty;
  c_state = rnd_state->c_state;
  memcpy(d_state, rnd_state->d_state, sizeof(d_state));
}
