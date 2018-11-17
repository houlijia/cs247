#ifndef __CC_SUM_MEAN_VAR_HDR__
#define __CC_SUM_MEAN_VAR_HDR__

/** \file
Compute sum, mean, var over blocks. Note that the vector is assumed to have
at least one elemement.
*/

#include <stddef.h>
#include <math.h>

#include "cc_vec_op_scalar.h"
#include "cc_sub_sqr.h"

/** Compute the sum of a vector of length > 0 in CPU */
template <class T>
T c_sum_vec(size_t n_vec,	/**< no. of elements in the block */
	    const T *vec     /**< the source vector of length \c n_vec. */
	    )
{
  T sum = vec[0];
  while(n_vec > 1)
    sum += vec[--n_vec];

  return sum;
}

/** Compute the mean of a vector of length > 0 in CPU */
template <class T>
T c_mean_vec(size_t n_vec,	/**< no. of elements in the block */
	     const T *vec     /**< the source vector of length \c n_vec. */
	     )
{
  return c_sum_vec(n_vec, vec)/T(n_vec);
}
     
/** Compute the mean of a vector of length > 0 in CPU */
template <class T>
T c_stdv_vec(size_t n_vec,	/**< no. of elements in the block */
	     const T *vec,     /**< the source vector of length \c n_vec. */
	     T mean	       /**<< The mean of the vector */
	     )
{
  T res = vec[0] * vec[0];
  T cnt = T(n_vec);

  while (n_vec-- > 1)
    res += vec[n_vec] * vec[n_vec];

  res -= cnt * mean * mean;
  
  return sqrt(res/(cnt - T(1)));
}
     

#endif	/*  __CC_SUM_MEAN_VAR_HDR__ */
