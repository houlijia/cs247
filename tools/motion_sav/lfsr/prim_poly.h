/* Copyright (C) 2012 Alcatel-Lucent */

/** \file prim_poly.h */

#ifndef _PRIM_POLY_H_
#define _PRIM_POLY_H_

#include <stddef.h>
#include "int_types.h"

/** A primitive polinomial is represented by an integer, where lower bits are
    coefficients of lower powers.  The highest coefficient (always 1) is
    omitted.
*/

typedef u_32 bitpol_t;

typedef struct PrimPoly {
  bitpol_t coefs;		/**< Generating polynomial */
  int degree;			/**< Degree of generating polynomial */

    /**  Maximal length sequence. Length is orderPrimPoly(prpl).
	 dynamically allocatred). Bits are ordered in each byte so
	 that lower index is less significant bits. if s(n),
	 n=0,1,2,.. are the elements of the sequence then the
	 generating rule is:
             s(n) = p(1)s(n-1)+...+p(r)*s(n-r)
         where r is the order of the polynomial and its coefficients
         are p(0)=1, p(1),..., p(r-1), p(r)=1. 
    */
  u_8 *seq;
} PrimPoly;

/** An array of PrimPoly objects */
typedef struct PrimPolyList{
  size_t cnt;			/**< number of PrimPoly objects */
  PrimPoly *prpl;		/**< Array of cnt objects (allocated by malloc */
} PrimPolyList;

#define PrimPolyListInitializer {0,NULL}

/** Type of a writing function which returns 0 if successful or an error code */
typedef int (*WH_write_fun)(bitpol_t L_val, bitpol_t R_val);

/** \return error on success or an error message in err_msg */
int readPrimPolyList(const char *fname,
		     PrimPolyList *prpl_lst,
		     char *err_msg, /**< An array for error message return */
		     size_t err_msg_size /**< size of err_msg */
		     );

#define orderPrimPoly(prpl) (((size_t)1 << (prpl)->degree)-1)

/** Compute maximum length sequence corresponding to polynomial
    \return a malloc allocated sequence of bytes (0 or 1) or NULL if an error
    occurred. In that case an error message is given err_msg.
*/
u_8 *comp_lfsrPrimPoly(const PrimPoly *prpl, /**< Primitive polynomial to use */
		       bitpol_t seed,        /**< initial state
						(non-zero) */
		       char *err_msg, /**< An array for error message return */
		       size_t err_msg_size /**< size of err_msg */
		       );

		
/** Generate and write out L() and R().
  \return 0 for success, else error code
 */
int comp_lfsr_wh(const PrimPoly *prpl, /**< Primitive polynomial to use. */
		 WH_write_fun wh_write /**< Write out sequence */
		 );	 

#endif	/* _PRIM_POLY_H_ */
