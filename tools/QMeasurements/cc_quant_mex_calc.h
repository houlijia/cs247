#ifndef __CC_QUANT_MEX_CALC_HDR__
#define __CC_QUANT_MEX_CALC_HDR__

#include "fast_heap.h"

template<class Float, class LBL>
static void calcCPU(size_t n_no_clip, //!< No. of measurements which are not clipped.
		    size_t n_clip, //!< No. of measurements which are to be clipped.
		    const Float *vec_no_clip, //!< An array of \c n_no_clip entries to quantize
		    const Float *vec_clip, //!< An array of \c n_clip entries to quantize
		    Float intvl,
		    Float offset,
		    LBL sat_lvl,
		    int nlhs,
		    mxArray *plhs[]
		    ) {
  plhs[0] = mxCreateNumericMatrix(n_no_clip, 1, mx_class_id<int32_T>(), mxREAL);
  plhs[1] = mxCreateNumericMatrix(n_clip, 1, mx_class_id_of(sat_lvl), mxREAL);

  GenericHeapElement *pghe = NULL;
  int32_T *save = NULL;
  size_t save_cnt;
  if(nlhs > 2) {
    GenericHeapElement &psave = fast_heap->get(n_clip * sizeof(int32_T));
    pghe = & psave;
    save = static_cast<int32_T *>(*psave);
  }
  save_cnt = c_quant(n_no_clip, n_clip, vec_no_clip, vec_clip, intvl, offset, sat_lvl,
		      (int32_T *) mxGetData(plhs[0]),
		      (LBL *) mxGetData(plhs[1]),
		      save);

  if(nlhs > 2) {
    plhs[2] = mxCreateNumericMatrix(save_cnt, 1, mx_class_id_of(save[0]), mxREAL);
    memcpy(mxGetData(plhs[2]), save, save_cnt*sizeof(*save));
    pghe->discard();
  }
}

#endif	/* __CC_QUANT_MEX_CALC_HDR__ */
