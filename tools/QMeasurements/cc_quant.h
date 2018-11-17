#ifndef __CC_QUANT_HDR__
#define __CC_QUANT_HDR__

/** File
This file contains functions for performing the operations of linear
quantization on the CPU. The functions are written as templates in  order to
allow both single and double precision implementations.

In the templates FLT (float or double)is the class of the measurments, QV is
the class of quantized labels (prior to saturation and LBL is the class of the
labels after saturation. 
 */

/** Quantization function. For each entry x the quntized value q is 
     <tt>    q = ceil((x+offset)/intvl) </tt>
    \c q is non-saturated if \c 0<q<sat_lvl and otherwise it is saturated. The
     coded labels output consists of three parts:
     \li \c q_no_clip - a vector containing the first \c n_no_clip quantized
     measurements, as is.
     \li \c q_vec - a vector containing the remaining \c n_vec-n_no_clip
     quantized measurements, where the value of each quantized measurements is
     set to \c sat_lvl.
     \li \c save (if not NULL) contains the original value of each of the
     saturated measurements in \c q_vec.
    \return the number of values put in \c save.
     \par EXAMPLE:
     Suppose the input vector was quantized in to -2,3,5,1,2,7, \c sat_lvl=4 and
     \c n_no_clip=2. Then \c q_no_clip will be -2,3, \c q_vec will be 4,1,2,4
     and
     \li if \c save is NULL the return value is 0.
     \li if \c save is non-NULL save will contain the values 5,7 and the
     return value is 2.

    \retrun the number of saturated values
 */
template<class FLT, class QV, class LBL>
size_t c_quant(size_t n_no_clip, //!< No. of measurements which are not clipped.
	       size_t n_clip, //!< No. of measurements which are to be clipped.
	       const FLT *vec_no_clip, //!< An array of \c n_no_clip entries to quantize
	       const FLT *vec_clip, //!< An array of \c n_clip entries to quantize
	       FLT intvl,      //!< The quantization interval
	       FLT offset,     //!< Quantizaton offset
	       LBL sat_lvl,	//!< Saturation level. If 0, no saturation
	       QV *q_no_clip,	 //!< A vector of length n_no_clip.
	       LBL *q_clip, //!< return: quantization labels (\c n_vec-n_no_clip elements)
	       QV *save   //!< return: saturated values (\c n_n_vec-n_no_clip elements)
	       )
{
  size_t j;
  size_t cnt = 0;

  for(j=0; j<n_no_clip; j++) {
    QV qval = (QV) ceil((vec_no_clip[j]+offset)/intvl - FLT(0.5));
    q_no_clip[j] = qval;
   }
  
  for(j=0; j<n_clip; j++) {
    QV qval = (QV) ceil((vec_clip[j]+offset)/intvl - FLT(0.5));

    if(qval < 1) {
      q_clip[j] = sat_lvl;
      if(save != NULL) 
	save[cnt++] = qval;
      }
    else if(qval >= (QV)sat_lvl) {
      q_clip[j] = sat_lvl;
      if(save != NULL) 
	save[cnt++] = qval - QV(sat_lvl-1);
      }
    else
      q_clip[j] = LBL(qval);
  }

  return cnt;
}

#endif	/* __CC_QUANT_HDR__ */
