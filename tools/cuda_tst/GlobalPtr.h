#ifndef __GlobalPtr_h__
#define __GlobalPtr_h__


/** \file
    class for saving and restoring mex context
*/

#include <list>
#include <string>

#ifdef MATLAB_MEX_FILE
#include "mex_context.h"
#endif

#ifndef ULONG_DEFINED
#define ULONG_DEFINED
typedef unsigned long ulong;
#endif

//* GlobalPtr<ObjT> is a smart pointer to an object of type ObjT, It
//* automatically creates the object (using \c new ObjT()) and destroys (witn
//* \c delete) it in its destructor. It also provides some operator -> for using
//* the object, but it protects the pointer from copying and
//* modification.
//*
//* Under MEX, this basic functionality is preserved but there are some
//* differences:
//*
//* - The ObjT object is not created in the constructor, but at the first time
//*   that operator -> is used.
//* - On that first time, if the object exists in the registry of MexCotnext,
//*   it is restored from there. Otherwise, it is created and saved in the registry.
//* - When the ObjT object is initialized, the pointer is copied and saved in
//*   the registry of ContextItem.
//* - When a Mex function is called and the context is restored, the value of
//*   pointer to ObjT is copied back from the MexItem registry.
//* - The ObjT object is not deleted by the destructor of
//*   GlobalPtr<ObjT>. Instead, it is deleted when the MexItem registry is
//*  destructed.

template<typename ObjT>
class GlobalPtr
{
public:
  //* Constructor
  //* \param ptr_name_: Under Mex, this is a pointer to the identifying name
  //* of this object, and is to be used by the registry. Without Mex, this
  //* parameter is ignored and can be omitted.
  //* Under Mex, initialization should be something like that:
  //* <tt>
  //* extern GlobalPtr<MyClass> xyz;  // usually from a aheader file.
  //*
  //* GlobalPtr<MyClass> xyz("xyz");
  //* </tt>
  GlobalPtr(const char *ptr_name_="")
    :
#ifdef MATLAB_MEX_FILE
  ptr(NULL), ptr_name(ptr_name_) {}
#else
  ptr(new ObjT) {(void)ptr_name_; }
#endif

  //* Destructor
  ~GlobalPtr()
#ifdef MATLAB_MEX_FILE
  {}
#else
  { delete ptr; }
#endif 

  ObjT * operator -> () 
    { init(); return ptr; }

private:
  //* forbid copy constructor
  GlobalPtr(const GlobalPtr &other);
 
  //* forbid assignment
  const GlobalPtr & operator = (const GlobalPtr &other);
  
private:			// data members
  //* Pointer to the object
  ObjT *ptr;

  void init()
  {
#if defined(MATLAB_MEX_FILE)
    if(ptr == NULL) MexContext::get(&ptr, ptr_name);
#endif
  }


#ifdef MATLAB_MEX_FILE
  //* Pointer to a function which restores the item to its place.
  std::string ptr_name;
#endif
};

#endif	// __GlobalPtr_h__
