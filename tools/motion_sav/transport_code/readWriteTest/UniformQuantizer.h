#ifndef UNIFOR_QUANTIZER_H
#define UNIFOR_QUANTIZER_H
#include "CodeElement.h"

class UniformQuantizer : public CodeElement
{
public:
  UniformQuantizer (const CodeElement& ce);

private:

  double
  calcIntvl (double q_unit)
  {
    if (this-> q_wdth_mltplr == 0 ||
        q_unit == 0)
      return 1;
    else if (q_unit == this->q_wdth_unit)
      return this->q_wdth;
    else
      return (this->q_wdth_mltplr * q_unit);
  }

  double q_wdth_mltplr;
  double q_ampl_mltplr;
  double q_wdth_unit;
  double q_wdth;
  bool save_clipped;

};

#endif
