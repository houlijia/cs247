#ifndef __cc_vec_op_binary_HDR__
#define __cc_vec_op_binary_HDR__

/** \file
Template function to perform element-wise binary operations on vectors. The different
argument vectors are epxected to be either disjoint or identical, but not paritally
overlapping.
*/

//! x -= y
template <class T>
void c_sub_asgn(size_t n_vec, //!< Vector length
				T *x,
				const T *y
				)
{
  for(size_t k=0; k<n_vec; k++) 
    x[k] -= y[k];
}

#endif	/* __cc_vec_op_binary_HDR__ */
