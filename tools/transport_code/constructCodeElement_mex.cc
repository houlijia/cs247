/**
   \file

   A mex function for for construction a C++ CodeElement object (or an object
   of a derived class).
*/

#include <string.h>

#include "CodeElement.h"

static CodeElement *newCodeElement(const mxArray *matlab_obj, const mxArray *mx_params) {
  return new CodeElement(matlab_obj, mx_params);
}

typedef CodeElement *NewCodeElement_T(const mxArray *matlab_obj, const mxArray *params);

typedef struct NameNew {
  const char *name;
  NewCodeElement_T *nce;
} NameNew;

static NameNew name_new[] = {
  {"CodeElement", &newCodeElement}
};

static const size_t n_name_new = sizeof(name_new)/sizeof(name_new[0]);

static char const * const errId = "constructCodeElement_mex:InvalidInput";

static const size_t MAX_NAME_LEN=256;

/** Arguments:
  Input:
    prhs[0] - pointer to the Matlab object which needs to be emulated
    prhs[1] - a struct specifying the contructor parameters.
  Output:
    plhs[0] - A uint8 array containing a pointer to the C++ object.

 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxArray *mx_spec;

  if(nrhs != 2 || nlhs != 1)
    mexErrMsgIdAndTxt(errId,
		       "constructCodeElement_mex should have 1 or 2 input "
		      "and one output arguments");
  
  if(!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
    mexErrMsgIdAndTxt(errId,
		      "Second argument should be a signle Matlab struct");
 
  const mxArray *mx_name = mxGetField(prhs[1], 0, "name");
  if (mx_name == NULL || !mxIsChar(mx_name) ||
      mxGetNumberOfElements(mx_name) > MAX_NAME_LEN)
    mexErrMsgIdAndTxt(errId,
		      "No \"name\" field, or it is not a string of up to %u chars",
		      (unsigned) MAX_NAME_LEN);

  char name[MAX_NAME_LEN+1];
  mxGetString(mx_name, name, sizeof(name));

  size_t k;
  CodeElementPtr dev_ptr;
  for(k=0; k<n_name_new; k++) {
    if(!strcmp(name, name_new[k].name)) {
       dev_ptr.ce = name_new[k].nce(prhs[0],prhs[1]);
       break;
    }
  }

  if(k==n_name_new)
    mexErrMsgIdAndTxt(errId, "Unrecongnized class: %s", name);
  else if (dev_ptr.ce == NULL)
    mexErrMsgIdAndTxt(errId, "Failed generating an object of class: %s", name);

  plhs[0] = mxCreateNumericMatrix(1,sizeof(dev_ptr), mxUINT8_CLASS, mxREAL);
  memcpy(mxGetData(plhs[0]), dev_ptr.b, sizeof(dev_ptr));
}
