/** \file
    Tools for working with MEX
*/

#include <string.h>

#include "mex.h"

#include "mex_tools.h"
#include "mex_assert.h"

size_t mxClassID_size(mxClassID cls_id)
{
  switch(cls_id) {
  case mxDOUBLE_CLASS: return sizeof(double);
  case mxSINGLE_CLASS: return sizeof(float);
  case mxINT64_CLASS:  return sizeof(int64_T);
  case mxUINT64_CLASS: return sizeof(uint64_T);
  case mxINT32_CLASS:  return sizeof(int32_T);
  case mxUINT32_CLASS: return sizeof(uint32_T);
  case mxINT16_CLASS:  return sizeof(int16_T);
  case mxUINT16_CLASS: return sizeof(uint16_T);
  case mxINT8_CLASS:   return sizeof(int8_T);
  case mxUINT8_CLASS:  return sizeof(uint8_T);
  case mxCHAR_CLASS:   return sizeof(char_T);
  default:
    mex_assert(false, 
	       ("mxClassID_size:Illegal_type", "Unexpected type: %d", int(cls_id)));
    return 0;
  }
}

void * mexGetPtrFromMatlab(const char *mtlb_var_name,
			   const char *mtlb_ws
			   )
{
  void *ptr;
  const mxArray *mxr = mexGetVariablePtr(mtlb_ws, mtlb_var_name);
  mex_assert(mxr != NULL, ("mexGetPtrFromMatlab:mexGetVariable_failed",
			   "No Matlab variable '%s' in workspace '%s'",
			   mtlb_var_name, mtlb_ws));

  size_t len_mxr = mxGetNumberOfElements(mxr);

  if(len_mxr == 0) {
    ptr = NULL;
  }
  else {

    mex_assert(mxGetClassID(mxr) == mxUINT8_CLASS && len_mxr == sizeof(ptr),
	       ("mexGetPtrFromMatlab:bad_ptr",
		"%s:%s should return a uint8 array (class id=%d) "
		"with sizeof(pointer)=%lu elements\n"
		"received class_id=%d with %lu elements",
		mtlb_ws, mtlb_var_name, int(mxUINT8_CLASS), ulong(sizeof(ptr)),
		int(mxGetClassID(mxr)), ulong(len_mxr)));

    memcpy(&ptr, mxGetData(mxr), len_mxr);

    mex_assert(ptr != NULL,
	       ("mexGetPtrFromMatlab:bad_ptr",
		"%s:%s returned is a NULL pointer", mtlb_ws, mtlb_var_name));
  }

  return ptr;
}

