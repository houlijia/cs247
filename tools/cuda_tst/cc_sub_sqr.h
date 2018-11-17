#ifndef __CC_SUB_SQR_HDR__
#define __CC_SUB_SQR_HDR__
/**
   Subtract a constant from an array and then square the elements of the array:
     res = (vec-sbval)^2.
  */

#include <stddef.h>


/** Same functionality as h_sub_sqr but without using GPU */
template <class T>
void c_sub_sqr(T sbval,	//!< Value to subtract
	       const T *vec,       //!< vector
	       size_t n_vec,	//!< No. of elements in \c vec
	       T *res	//!< Result vector
	)
{
  size_t i;

  for(i=0; i<n_vec; i++) {
    T dff = vec[i] - sbval;
    res[i] = dff * dff;
  }
}

#endif /*  __CC_SUB_SQR_HDR__ */
