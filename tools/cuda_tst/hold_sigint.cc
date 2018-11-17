/** \file */

#include <assert.h>
#include <stdio.h>
#include <signal.h>

#include "hold_sigint.h"
#include "mex_tools.h"
#include "mex_assert.h"

unsigned HoldSigInt::level = 0;
bool HoldSigInt::got_sig = false;
void (* HoldSigInt::prev_handler)(int);

void HoldSigInt::sigint_handler(int sig)
{
  assert(sig == SIGINT);

  got_sig = true;
}

HoldSigInt::HoldSigInt()
{
  if(level++ == 0) {
    prev_handler = signal(SIGINT, HoldSigInt::sigint_handler);
    assert(prev_handler != SIG_ERR);
  }
}

HoldSigInt::~HoldSigInt()
{
  assert(level > 0);
  if(--level == 0) {
    bool flag = got_sig;
    got_sig = false;
    signal(SIGINT, prev_handler);
    if(flag) {
#ifdef MATLAB_MEX_FILE
      mexPrintf("\n ****** Received SIGINT ******\n");
      mexErrMsgIdAndTxt("Signal:SIGINT", " ****** Received SIGINT ******\n");
#else
      fprintf(stderr, "received SIGINT\n");
#endif
      raise(SIGINT);
    }
  }
}
