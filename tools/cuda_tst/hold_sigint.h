/** \file */

#ifndef __hold_siging_h__
#define __hold_siging_h__

class HoldSigInt {
public:

  HoldSigInt();
  ~HoldSigInt();
private:
  static unsigned level;
  static bool got_sig;
  static void (*prev_handler)(int);
  static void sigint_handler(int);
};
#endif	/* __hold_siging_h__ */
