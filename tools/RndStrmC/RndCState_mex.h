#ifndef RndCState_mex_h
#define RndCState_mex_h

#include <mex.h>
#include <matrix.h>

struct RndCState;

extern const char *RndCState_flds[];

mxArray *RndCState_to_mex(const struct RndCState *rnd_state);
void RndCSTate_from_mex(struct RndCState *rnd_state, const mxArray *mx_rnd_state);


#endif	/* RndCState_mex_h */
