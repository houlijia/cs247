#ifndef __mex_context_h__
#define __mex_context_h__


/** \file
   class for saving and restoring mex contextg
 */

#ifdef MATLAB_MEX_FILE
#include <string>
#include <map>

#include "mex.h"

#include "mex_assert.h"
#include "mex_tools.h"

#include "Mutex.h"

#ifndef ULONG_DEFINED
#define ULONG_DEFINED
typedef unsigned long ulong;
#endif
 
class MexContext 
{
public:
  static const MexContext* getContext();
  static const MexContext* resetContext();

  template<typename ObjT>
  static void get(ObjT **pp, const std::string &name)
  { setContext();
    mex_context->getItem<ObjT>(pp, name);
  }

  static void deleteContext();

private:

  class GenericItem
  {
  public:
    GenericItem() {}
    GenericItem(const GenericItem &other) {(void) other; }
    virtual ~GenericItem() {};
   };

  template<typename ObjT>
  class TypedItem : public GenericItem
  {
  public:
    TypedItem(ObjT *p=NULL) : ptr(p) {}

    //* Copy constructor - move ownership of pointer to *this
    TypedItem(TypedItem<ObjT>& other): ptr(other.ptr) {other.ptr = NULL; }
   
    virtual ~TypedItem() {}

    //* assignment operator - move ownership of pointer to *this
    const TypedItem& operator=(TypedItem<ObjT>& other)
    {ptr = other.ptr; other.ptr = NULL; return *this; }

    ObjT *getPtr() const
    { return ptr; }
  private:
    ObjT* ptr;
  };
  
  typedef std::map<std::string, GenericItem* > backup_t;

  MexContext();

  ~MexContext();

  template<typename ObjT>
  void getItem(ObjT **pp, const std::string &name);

  static void setContext()
  {
    if(mex_context == NULL)
      mex_context = static_cast<MexContext *>(mexGetPtrFromMatlab("mex_context_val"));
  }
  		    
  //* Associative array of backup of context items, indexed by name
  backup_t backup;

  Mutex mutex;

  static MexContext *mex_context;
};

template<typename ObjT>
void MexContext::getItem(ObjT **pp, const std::string &name) {

  MutexLock<Mutex> lock(mutex);

  backup_t::iterator pos = backup.find(name);
  if(pos == backup.end()) {
    // not in backup - must be created
    *pp = new ObjT();
    GenericItem *gi = new TypedItem<ObjT>(*pp);
    backup.insert(backup_t::value_type(name, gi));
  } else {
    // found in backup, \c pos points to it.
    GenericItem& gi = *(pos->second);
    *pp = dynamic_cast<TypedItem<ObjT>&>(gi).getPtr();
  }
}

#endif	// def MATLAB_MEX_FILE

#endif	// __mex_context_h__
