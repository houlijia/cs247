#ifndef __ObjectPool_h__
#define __ObjectPool_h__

/** \file
    This file contains template for generating a pool of objects.
 */ 


#include <stddef.h>
#include <errno.h>
#include <string.h>
#include <stack>
#include <vector>
#include <exception>

#include "mex_assert.h"
#include "Mutex.h"

// Forward declaration
class GenericObjectPool;

//* A base class for any type of object stored in ObjectPool. Its main purpose
//* is to provide the discard() function
struct GenericPoolObject {
  GenericPoolObject(GenericObjectPool * const pool_)
    : pool(pool_) {}

  //* Destructor. \note NOT Virtual: I do not expect to see pointers to
  //* GenericPoolObject 
  ~GenericPoolObject() {};

  void discard();

  GenericObjectPool * const pool;

private:
  // No default constructor, copy constructor or assignement
  GenericPoolObject();
  GenericPoolObject(const  GenericPoolObject&);
  const GenericPoolObject& operator= (const  GenericPoolObject&);
};

//* A base class of ObjectPool. Its main purpose is to provide the pure
//* virtual function discard().
class GenericObjectPool {
public:
  GenericObjectPool()
    : n_out(0) {};
  
  virtual void discard(GenericPoolObject *obj) = 0;

 //* Return the number of checked out objects
  size_t nOut() const { return n_out; }

  //* delete all unused elements
  virtual void clear() = 0;
  
  
protected:
  virtual ~GenericObjectPool() {};

  size_t n_out;			// Number of objects checked out.
  
};

inline void GenericPoolObject::discard()  { pool->discard(this); }

//* This is a template of a pool of object of type ObjType. ObjectPool has two
//* main functions: get() and put(). \c get() returns a reference to an available
//* object of type ObjType and\c  put() returns an object to the pool. In
//* addition, the \c size() function reports number of objects available in
//* the pool.
//*
//* If get() is called and the pool is empty, get() creates a new object and
//* returns it. Therefore,the number of objects that are created is the
//* maximum number of objects that may be in use concurrently.
//*
//* When get() creates an object, it uses a constructor that gets one argument
//* of ObjTypeSpec.
//* 
//* empty() returns true if the pool is empty.
//* size() returns the number of elements in the pool
//* elmntSize() returns spec.size()
//* 
//* Assumptions about ObjType and ObjTypeSpec:
//*   ObjType has public constructor 
//*        ObjType(const ObjTypeSpec &, GenericObjectPool *)
//*   ObjTypeSpec has a public copy constructor ObjTypeSpec(const ObjTypeSpec &)


template<class ObjType, class ObjSpecType, class mutex_t = Mutex>
class ObjectPool: public ObjSpecType, public GenericObjectPool
{
public: 
  //* Constructor
  ObjectPool(const ObjSpecType &spec, mutex_t *pmtx=NULL)
    : ObjSpecType(spec), mtx_allocated(pmtx==NULL),
      mtx(*(pmtx==NULL? new mutex_t: pmtx)) {}

  //* Destructor
  ~ObjectPool() {
    clear();
    if(mtx_allocated) delete &mtx;
  }

  //* delete all unused elements
  void clear() {
    mtx.lock();

    // destroy all objects in the stack
    while(!stk.empty()) {
      delete stk.top();
      stk.pop();
    }

    mtx.unlock();
  }
  
  ObjType & get() {
    ObjType *pobj;

    mtx.lock();
    if(stk.empty()) {
      ++n_out;
      mtx.unlock();
      pobj = (ObjType *) new ObjType(*this, this);
      mex_assert(pobj != NULL, 
		 ("ObjectPool:get:alloc", "ObjectPool::get() failed to allocate an object"));
    } else {
      pobj = stk.top();
      stk.pop();
      ++n_out;
      mtx.unlock();
    }
    return *pobj;
  }

  void put(ObjType &robj) {
    mtx.lock();
    stk.push(&robj);
    mex_assert(n_out > 0,
	       ("ObjectPool:put:n_out", "put called when n_out is 0"));
    --n_out;
    mtx.unlock();
  }

  void discard(GenericPoolObject *obj) { put(*(ObjType *)obj); }

  bool empty() const { 
    mtx.lock();
    bool result = stk.empty();
    mtx.unlock();
    return result;
  }

  size_t size() const { 
    mtx.lock();
    size_t result = stk.size();
    mtx.unlock();
    return result;
  }

  size_t elmntSize() const { return this->ObjSpecType::size(); }

private:
  std::stack<ObjType *, std::vector<ObjType *> > stk;
  bool mtx_allocated;
  mutex_t &mtx;
};				// ObjectPool

#endif	/*  __ObjectPool_h__ */
