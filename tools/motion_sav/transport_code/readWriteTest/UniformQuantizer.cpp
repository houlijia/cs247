#include "UniformQuantizer.h"
#include "readUInt.h"
#include "readRealNumber.h"
#include <iostream>

using std::cout;
using std::endl;

UniformQuantizer::UniformQuantizer (const CodeElement& ce) : CodeElement (ce)
{
  const char* errString = NULL;
  int endP = 0;
  uint32 nBytes = 0;
  char* currentP;
  longInt sv_clp = readOnlyOneUInt (this->data, 0, &endP, &errString);
  if(sv_clp)
    this->save_clipped = true;
  else

    this->save_clipped = false;
  currentP = this->data + endP + 1;

  this->q_wdth_mltplr = readOnlyOneRealNumber (currentP, &nBytes, &errString);
  currentP += nBytes;
  double q_unit = readOnlyOneRealNumber (currentP, &nBytes, &errString);

  this->q_wdth = calcIntvl (q_unit);

  currentP += nBytes;
  this->q_ampl_mltplr = readOnlyOneRealNumber (currentP, &nBytes, &errString);
  this->q_wdth_unit = q_unit;

  cout << endl << "@@:" << this->save_clipped << "  " << q_wdth_mltplr << "  " << q_unit << "  " << q_ampl_mltplr << "  " << q_wdth << endl;
}


