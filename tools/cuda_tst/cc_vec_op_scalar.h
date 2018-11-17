#ifndef __CC_VEC_OP_SCALAR_HDR__
#define __CC_VEC_OP_SCALAR_HDR__

/** \file
Functions to perform operations between a vector and a scalar. The operations
are performed between each entry of the vector and the scalar and the results
are returned in the same vector. The scalar may be supplied as a pointer to a
value on the GPU or as an argument.
*/

#include <stddef.h>

/** Divide a vector \c vec of length \c n_vec by \c scalar and
 return result in \c res
*/
template <class T>
void 
c_vec_div_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  size_t k;
  for (k=0; k<n_vec; k++)
    res[k] = vec[k] / scalar;
}

/** Divide a vector \c vec of length \c n_vec by \c scalar and
 return result in place
*/
template <class T>
void 
c_vec_div_scalar(T scalar,
		 size_t n_vec,
		 T* vec
		 )
{
  size_t k;
  for (k=0; k<n_vec; k++)
    vec[k] = vec[k] / scalar;
}

/** Multiply a vector \c vec of length \c n_vec by \c scalar and
 return result in \c res
*/
template <class T>
void 
c_vec_mlt_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  size_t k;
  for (k=0; k<n_vec; k++)
    res[k] = vec[k] * scalar;
}

/** Add a scalar \c scalar to a vector \c vec of length \c n_vec and
 return result in \c res
*/
template <class T>
void 
c_vec_add_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  size_t k;
  for (k=0; k<n_vec; k++)
    res[k] = vec[k] + scalar;
}

/** Subtract a scalar \c scalar to a vector \c vec of length \c n_vec and
 return result in \c res
*/
template <class T>
void 
c_vec_sub_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  size_t k;
  for (k=0; k<n_vec; k++)
    res[k] = vec[k] - scalar;
}



#endif	/*  __CC_VEC_OP_SCALAR_HDR__ */
