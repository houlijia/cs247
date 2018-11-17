#ifndef __CodeElement_HDR__
#define __CodeElement_HDR__

/** \file

The behavior of this file and files of derived classes depends on some macros
which are assumed to be coming from the outside:

- MATLAB_MEX_FILE is true if the filed is compiled in a MEX environment
- HAS_GPU is true if CUDA code needs to be generated as well
 */

/** A class corresponding to CodeElement. Currently mostly a stub. */

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

class CodeElement {
public:
  CodeElement() {
#ifdef MATLAB_MEX_FILE
    this->matlab_obj = NULL;
#endif
  };
  
#ifdef MATLAB_MEX_FILE
  CodeElement(const mxArray *mtlb_obj, const mxArray *mx_params){
    this->matlab_obj = mtlb_obj;
    (void) mx_params;
  }
#endif

  ~CodeElement() {};

protected:
#ifdef MATLAB_MEX_FILE
  const mxArray *matlab_obj;
#endif

};

#ifdef MATLAB_MEX_FILE
typedef union CodeElementPtr {
  unsigned char b[1];
  CodeElement *ce;
} CodeElementPtr ;
#endif



#endif	/* __CodeElement_HDR__ */
