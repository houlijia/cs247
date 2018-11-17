#ifndef __MEX_TOOLS_HDR__
#define __MEX_TOOLS_HDR__

#ifdef MATLAB_MEX_FILE

#include "tmwtypes.h"
#include "mex.h"

#define mxIsScalar(x) (mxGetNumberOfDimensions(x)==2 && mxGetM(x)==1 && mxGetN(x)==1)
#define mxIsVector(x) (mxGetNumberOfDimensions(x)==2 && (mxGetM(x)==1 || mxGetN(x)==1))
#define mxIsString(x) (mxIsChar(x) && mxGetNumberOfDimensions(x)==2 && mxGetM(x)<=1)

#ifndef ULONG_DEFINED
#define ULONG_DEFINED
typedef unsigned long ulong;
#endif

#ifdef __cplusplus

//* Convert type to mxClassID
template <class T>
inline mxClassID mx_class_id(void) {
  mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput","Unexpected class ID");
  return mxUNKNOWN_CLASS;
}

template <> inline mxClassID mx_class_id<double>(void) { return mxDOUBLE_CLASS; }
template <> inline mxClassID mx_class_id<float>(void) { return mxSINGLE_CLASS; }
template <> inline mxClassID mx_class_id<int64_T>(void) { return mxINT64_CLASS; }
template <> inline mxClassID mx_class_id<uint64_T>(void) { return mxUINT64_CLASS; }
template <> inline mxClassID mx_class_id<int32_T>(void) { return mxINT32_CLASS; }
template <> inline mxClassID mx_class_id<uint32_T>(void) { return mxUINT32_CLASS; }
template <> inline mxClassID mx_class_id<int16_T>(void) { return mxINT16_CLASS; }
template <> inline mxClassID mx_class_id<uint16_T>(void) { return mxUINT16_CLASS; }
template <> inline mxClassID mx_class_id<int8_T>(void) { return mxINT8_CLASS; }
template <> inline mxClassID mx_class_id<uint8_T>(void) { return mxUINT8_CLASS; }
template <> inline mxClassID mx_class_id<char_T>(void) { return mxCHAR_CLASS; }

//* Get mxClassID of a variable
template <class T>
inline mxClassID mx_class_id_of(T dummy) {
  (void) dummy;
  return mx_class_id<T>();
}

//* \return sizeof the specified type
size_t mxClassID_size(mxClassID cls_id);

/* 
 Read a pointer to a MEX object (A C/C++/Cuda style pointer from a Matlab
 work space. The pointer should have been saved there previously as an
 array of uint8 (bytes).
 \param mtlb_var_name Matlab variable name
 \param mtlb_ws Matlab workspace in which variable is defined. can be
 "global" (default), "caller" (calling function workspace) or "base" (base
 workspace).
 \return pointer value.
*/
void * mexGetPtrFromMatlab(const char *mtlb_var_name,
			   const char *mtlb_ws = "global"
			   );

#endif	/* __cplusplus */


#include "mex.h"
#define printf mexPrintf

#endif	/* #ifdef MATLAB_MEX_FILE */

#endif	/* __MEX_TOOLS_HDR__ */
