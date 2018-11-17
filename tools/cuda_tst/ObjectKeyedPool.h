#ifndef __ObjectKeyedPool_h__
#define __ObjectKeyedPool_h__

/** \file 
    This file contains template for generating an array of pools
    of objects of different sizes.

*/ 

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <map>

#include "ObjectPool.h"
#include "mex_assert.h"
#include "mex_tools.h"

//* This is a template for an associative array of ObjectPool pools (see
//* ObjectPool.h). The keys, or indices to the associative array are the
//* object specifications of type ObjSpecType. Accordingly, A
//* ObjectKeyedPool contains a STL map of type
//*    map<const ObjSpecType, ObjectPool<ObjType, ObjSpecType>* >
//*
//* The main methods are
//* - get(ObjSpecType &spec) gets an object with the minimal size that fits the need
//* - put returns the object to the correct pool.
//* - empty() returns true if no pools are defined.
//* - size() returns the number of pools.
//* - total() returns sum of size of all pools
//* - nOut() returns the number of objects checked out from all pools 
//*
//* The assumptions on ObjTypeSpec and ObjType are the same as in ObjectPool.h
//*
//* ObjectPool objects are created on demand. ObjectKeyedPool has a mutex and
//* each pool has its own mutex.

template <class ObjType, class ObjSpecType, class mutex_t = Mutex>
class ObjectKeyedPool
{
  // Using the alias pool_t for ObjectPool<ObjType, ObjSpecType>
  //  would make the code more readable. However, with gcc ver. 4.9.3 on
  //  Cygwin, neither of the following "using" or "typedef" commands were
  //  accepted. Hence we used the explicit templated type name.
  //
  //  using pool_t = ObjectPool<ObjType, ObjSpecType>;
  //  typedef ObjectPool<ObjType, ObjSpecType> pool_t;

  typedef ObjectPool<ObjType, ObjSpecType> pool_t;
  typedef std::map<const ObjSpecType, pool_t > pools_t;
public:
  ObjectKeyedPool() {}

  ~ObjectKeyedPool() {}

  //* Delete all unused elements
  void clear() { 
    MutexLock<Mutex>(this->mutex);
    for(typename pools_t::iterator p=pools.begin(); p!=pools.end(); ++p) 
      p->second.clear(); 
  }

  size_t nOut() const {
    size_t n=0;
    MutexLock<Mutex>(this->mutex);
    for(typename pools_t::iterator p=pools.begin(); p!=pools.end(); +p)
      n += p->second.nOut();
    return n;
  }
  
  //* Verify that no objects are out and then perform clear()
  void reset() {
    mex_assert(nOut() == 0, ("ObjectKeyedPool:reset", 
			     "%lu objects are out", (unsigned long) nOut()));
    clear();
  }

      
  ObjType & get(const ObjSpecType& spec) {
    MutexLock<Mutex> mtx(this->mutex);
    typename pools_t::iterator pos = pools.find(spec);
    if(pos == pools.end())
      pos = pools.insert(typename pools_t::value_type(spec,pool_t(spec, &mutex))).first;
    return pos->second.get();
  }
 
  void put(ObjType &robj) {
    pool_t *pl = static_cast<pool_t *>(robj.GenericPoolObject::pool);
    pl->put(robj);
  }

  bool empty () const { 
    MutexLock<Mutex>(this->mutex);
    return pools.empty(); 
  }

  size_t size() const { 
    MutexLock<Mutex>(this->mutex);
    return pools.size(); 
  } 

  size_t total() const {
    size_t ttl = 0;
    MutexLock<Mutex>(this->mutex);
    for(typename pools_t::iterator p=pools.begin(); p!=pools.end(); +p)
 	ttl += p->second.size();
    return ttl;
  }

private:

  mutable Mutex mutex;
  pools_t pools;
  
};

#endif	/*  __ObjectKeyedPool_h_- */
