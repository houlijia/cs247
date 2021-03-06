/* Copyright (C) 2012 Alcatel-Lucent */

/** \file comp_lfsr.c 

  General theory:
  ==============

  Let

    f[x] = p(0)+p(1)x+...+p(r)x^r
     
  be a generating primitive polynomial. Let 

    x^(-j) = g(j)[x] = g(j,0)+g(j,1)x+...g(j,r-1)x^(r-1) (mod f[x])

  where g(j)[x] is a polynomial of degree r-1. g(j)[x] is unique. In
  particular, from the definition of f[x] and since p(0)=1:

    g(1)[x] = p(1)+p(2)x+...+p(r)x^(r-1), or equivalently
    g(1,i) = p(i+1) for 0<=i<r

  Therefore we have the recusions 

    g(j+1)[x] = g(j,0)*g(1)[x] + g(j,1) + g(j,2)x+...+g(j,r-1)x^(r-2)
 
  hence

    g(j+1,i) = g(j,i+1) + g(j,0)*p(i+1)

  where g(j,r) is taken as zero. Similarly,

    g(j-1,i) = g(j,i-1) + g(j,r-1)*p(i)

  where g(j,-1)is taken as zero.

  Note also that 

    g(-j)[x] = x^j or equivalently g(-j,i) = TRUE(i==-j),  for 0<=j<r

  Let s(n) be a maximum length sequence generated by f[x], that is,  

    s(n) = p(1)s(n-1)+...+p(r)*s(n-r)

  It can easily be shown that

    s(n+j) = g(j,0)s(n)+...+g(j,r-1)s(n-(r-1))  for j>-r+1

  Note also that 

     0 = x^(-n) f[x] = p(0)x(-n)+...+p(r)x^(-n+r) =
         p(0)g(n)[x]+...+p(r)g(n-r)[x] (mod f[x])

  Hence for any 0<=i<r, the seqence g(n,i), n=0,...,2^r-1 is a maximum
  length sequence generated by f[x].  Different values of i give different
  shifted version of the same sequence.

  Let H be a Walsh-Hadamard matrix of order 2^r.  Assume that the rows of H
  are in Hadamard order, that is, H is the Kroenecker product of r matrices of
  the form
      1  1
      1 -1

  then s(i-j) = H(L(i),R(j)) (indexing of H starts from zero) where

    L(i) = s(i)+s(i-1)*2+...+s(i-r+1)*2^(r-1)
    R(j) = g(-j)[2]

  This is based on: A. Cohn and A. Lempel, "On Fast M-Sequence Transform",
  IEEE Trans. Information Theory, vol. 23(1), Jan. 1977, pp. 135-137".
  
*/

#include <stdio.h>
#include <errno.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

#include "prim_poly.h"

/** Compute one step iteration of computing a maximum length seqeuce s(j).
    \param[in] status - the g(j)[x] polynomial
    \param[in] pol - the g(1)[x] polynomial, p(1)+p(2)x+...+p(r)x^(r-1)
    \param[in,out] seq - byte sequence of bits of ML sequence
    \param[in,out] out_mask - position in byte where new sequence value is
    placed.
    \param[in,out] out_indx - index of byte where new sequence value is
    placed.
    \return g(j+1)[x] polynomial
 */
static bitpol_t comp_next(bitpol_t status, bitpol_t pol, 
			  u_8 *seq, u_8 *out_mask, size_t *out_indx)
{
    u_8 bit = (u_8)(status & 1);

    status = status >> 1;
    if(bit) {
	seq[*out_indx] |= *out_mask;
	status = status ^ pol;
    }

    *out_mask <<= 1;
    if(!*out_mask) {
	*out_mask = 1;
	(*out_indx)++;
    }

    return status;
}

/** Comute a ML sequence starting with a seed - an initial value for g(j)[x].
    The sequence is the sequence g(j,0).
*/
 u_8 *comp_lfsrPrimPoly(const PrimPoly *prpl, /**< Primitive polynomial to use */
		       bitpol_t seed,        /**< initial state (non-zero) */
		       char *err_msg, /**< An array for error message return */
		       size_t err_msg_size /**< size of err_msg */
		       )
{
    u_8 *seq;
    size_t order = orderPrimPoly(prpl);
    size_t k;

    /** pol is the polynomial g(1)[x] */
    bitpol_t pol = (prpl->coefs >> 1) | (1<<(prpl->degree-1)); 

    bitpol_t status;    /**< Current value of b(j)[x] */
    u_8 out_mask = 1;
    size_t out_indx = 0;

    /* Using calloc() to initialize to zero */
    seq = (u_8 *) calloc((order+1)/8, sizeof(u_8));
    if(seq == NULL) {
	snprintf(err_msg, err_msg_size, 
		 "Failed to allocate 0x%lX bytes for degree %ld sequence",
		 (unsigned long) order, (unsigned long)prpl->degree);
	return seq;
    }

    status = comp_next(seed, pol, seq, &out_mask, &out_indx);
 
    for(k=1; k<order; k++) {
	if(status == seed) {
	    snprintf(err_msg, err_msg_size, "status repeated after %lu values in degree %ld",
		     (unsigned long) k, (unsigned long) prpl->degree);
	    free(seq);
	    return NULL;
	}

	status = comp_next(status, pol, seq, &out_mask, &out_indx);
    }

    return seq;
}

static bitpol_t updateLval(bitpol_t L_val,
			   const u_8 *seq,
			   size_t *seq_indx,
			   bitpol_t seq_high,
			   size_t period
			   )
{
    size_t indx;
    u_8 mask;

    mask = 1<<(*(seq_indx) & 7);
    indx = (*seq_indx) >> 3;
    *seq_indx += 1;
    if (*seq_indx >= period)
	*seq_indx = *seq_indx - period;

    L_val &= ~seq_high;
    L_val = L_val <<= 1;
    L_val |= ((seq[indx] & mask) != 0);
    return L_val;
}			      

/** Generate and write out L() and R().
 */
int comp_lfsr_wh(const PrimPoly *prpl, /**< Primitive polynomial to use. seq
					  must be already computed */
		 WH_write_fun wh_write /**< Write out sequence */
		 )
{
    u_32 L_val=0, R_val=1;
    size_t k;
    size_t period = (1<<(prpl->degree)) - 1;
    size_t seq_indx = period - prpl->degree + 1;
    const int rm1 = (prpl->degree) - 1;
    const u_32 seq_high = (u_32)(1<<rm1);
    int err;

    /* Initialize L_val */
    for (k=0; k<prpl->degree; k++)
	L_val = updateLval(L_val, prpl->seq, &seq_indx, seq_high, period);

    err = wh_write(L_val, R_val);

    for (k=1; !err && k<period; k++) {
	L_val = updateLval(L_val, prpl->seq, &seq_indx, seq_high, period);

	/* Update R_val */
	if (R_val & seq_high) {
	    R_val ^= seq_high;
	    R_val = (R_val<<1) ^ prpl->coefs;
	} else
	    R_val = R_val<<1;

	err = wh_write(L_val, R_val);
    }

    return err;

}
    

    
