#ifndef __CC_REAL_DFT_SORT_HDR__
#define __CC_REAL_DFT_SORT_HDR__

/**
   \file

   function to convert between DFT output of a real signal to a more compact
   form. Let \c N be the DFT order and let x[0],...,x[N-1] be the complex DFT
   coefficients. If N is even, the compact form is:

   Re{x[0]},Re{x[N/2]},Re{x[1]},Im{x[1]},...,Re{x[N/2-1]},Im{x[N/2-1]}

   If N>1 is odd the compact form is 

   Re{x[0]},Re{x[1]},Im{x[1]},...,Re{x[(N-1)/2]},Im{x[(N-1)/2]}

   If N=1 the compact form is

   Re{x[0]}

   The functions perform the conversions on the columns of matrices with M
   columns.

*/

//! Convert DFT coefficients to compact form
template<class Float>
void c_real_dft_sort(size_t N,	//!< DFT order
		     size_t M,  //!< Number of columns (>0)
		     const Float *rl, //!< Real part of coefficients (>0)
		     const Float *im, //!< Imaginary part of coefficients
		     Float *cmpct     //!< Output array of size M*N
		     )
{
  size_t j;
  
  if(N%2) {
    for(j=0; j<M; j++) {
      size_t n_cl = (N-1)/2;
      Float *cmpct_cl = cmpct + j*N;
      const Float *rl_cl = rl + j*N;
      const Float *im_cl = im + j*N;

      *cmpct_cl++ = *rl_cl++;
      im_cl++;
  
      while(n_cl--) {
	*cmpct_cl++ = *rl_cl++;
	*cmpct_cl++ = *im_cl++;
      }
    }
  }
  else {
    for(j=0; j<M; j++) {
      size_t n_cl = N/2;
      Float *cmpct_cl = cmpct + j*N;
      const Float *rl_cl = rl + j*N;
      const Float *im_cl = im + j*N;

      *cmpct_cl++ = *rl_cl++;
      im_cl++;
      *cmpct_cl++ = rl_cl[--n_cl];
  
      while(n_cl--) {
	*cmpct_cl++ = *rl_cl++;
	*cmpct_cl++ = *im_cl++;
      }
    }
  }
}

//! Convert compact form to DFT coefficients
template<class Float>
void c_real_dft_unsort(size_t N,	//!< DFT order
		       size_t M,  //!< Numer of columns (>0)
		       const Float *cmpct,     //!< input array of size N (>0)
		       Float *rl, //!< Output: Real part of coefficients (size N)
		       Float *im //!< Output: Imaginary part of coefficients
				  //!(size N)
		       )
{
  size_t j,k;
  size_t N2=N/2;
  
  if(N%2 == 0) {
    for(j=0; j<M; j++) {
      const Float *cmpct_cl = cmpct + j*N;
      Float *rl_cl = rl + j*N;
      Float *im_cl = im + j*N;

      rl_cl[0] = *cmpct_cl++;
      im_cl[0] = 0;
      rl_cl[N2] = *cmpct_cl++;
      im_cl[N2] = 0;

      for(k=1; k<=N2-1; k++) {
	rl_cl[k] = rl_cl[N-k] = *cmpct_cl++;
	im_cl[k] = *cmpct_cl++;
	im_cl[N-k] = -im_cl[k];
      }
    }
  }
  else {
    for(j=0; j<M; j++) {
      const Float *cmpct_cl = cmpct + j*N;
      Float *rl_cl = rl + j*N;
      Float *im_cl = im + j*N;
      size_t N2=N/2;

      rl_cl[0] = *cmpct_cl++;
      im_cl[0] = 0;
 
      for(k=1; k<=N2; k++) {
	rl_cl[k] = rl_cl[N-k] = *cmpct_cl++;
	im_cl[k] = *cmpct_cl++;
	im_cl[N-k] = -im_cl[k];
      }
    }
  }
}

#endif	/*  __CC_REAL_DFT_SORT_HDR__ */
