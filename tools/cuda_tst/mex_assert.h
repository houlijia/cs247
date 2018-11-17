#ifndef __MEX_ASSERT_HDR__
#define __MEX_ASSERT_HDR__

/** The macro mex_assert receives two arguments:
    tst - a variable which is tested for being true
    args - an expression in parenthesis which is the arguments to
    mexErrMsgIdAndTxt()

    The operation of this macro depends on the macros NDEBUG and
    MATLAB_MEX_FILE:

    if NDEBUG is defined and true the macro is not evaluated at all
    else if MATLAB_MEX_FILE is defined and true, \c tst is evaluated and if
    false, mexErrMsgIdAndTxt() is called with \c args
    else \c tst is evaluated and if false the program is aborted with an 
    failure message pointing to the file and line of the error.

    \note Unlike standard assert, it is disabled only if NDEBUG is true (being
    defined is not enough).

*/

#include <stdarg.h>


#if defined(NDEBUG) && NDEBUG

inline bool mex_assert_print(const char *err_id,
			       const char *fmt,
			       ...)
{
  (void) err_id; (void) fmt;
  return true;
}


#define mex_assert(tst,args) mex_assert_print args

#elif defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE

#include "mex.h"

inline bool mex_assert_print(const char *err_id,
			       const char *fmt,
			       ...)
{
  mexPrintf("%s ", err_id);
  va_list ap; va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  return true;
}

#define mex_assert(tst,args) ((tst) || \
			      (mexPrintf("**** %s:%d mex_assert failed! ****\n", __FILE__, __LINE__), \
			       mex_assert_print args , mexErrMsgIdAndTxt args ,1))

#else

#include <stdio.h>
#include <stdlib.h>

inline bool mex_assert_print(const char *err_id,
                             const char *fmt,
                             ...) {
  fprintf(stderr, "%s ", err_id);
  va_list ap; va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  abort();
  return true;
}

#define mex_assert(tst,args) ((tst) ||					\
			      (fprintf(stderr,"**** %s:%d mex_assert failed! ****\n", \
				       __FILE__, __LINE__), mex_assert_print args , abort(), 1))

#endif

#endif	/* __MEX_ASSERT_HDR__ */
