#ifndef __CC_VEC_OP_UNARY_HDR__
#define __CC_VEC_OP_UNARY_HDR__

/** \file
Functions to perform unary operations on vectors, entry by entry.
*/

#include <stddef.h>

/** take square root of a vector \c vec of length \c n_vec and return results in res
    (res can be the same as vec). Coputation is on CPU */
template <class T>
void
c_vec_sqrt(const T *vec,
	   size_t n_vec,
	   T *res
	   )
{
  while(n_vec-- > 0)
    res[n_vec] = T(sqrt(vec[n_vec]));
}

/** take aboslute value of a vector \c vec of length \c n_vec and return results in res
    (res can be the same as vec). Coputation is on CPU */
template <class T>
void
c_vec_abs(const T *vec,
	  size_t n_vec,
	  T *res
	  )
{
  while(n_vec-- > 0)
    res[n_vec] = (vec[n_vec]<0)? -vec[n_vec]: vec[n_vec];
}

#endif	/*  __CC_VEC_OP_UNARY_HDR__ */
