#include <stdio.h>
#include <unistd.h>
#include "hold_sigint.h"

int main()
{
  HoldSigInt hold_sig_int;

  for(int k=5; k>0; --k) {
    printf("Sleeping...\n");
    sleep(2);
  }
  printf("Done\n");
}
