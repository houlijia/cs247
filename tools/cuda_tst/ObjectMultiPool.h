#ifndef __ObjectMultiPool_h__
#define __ObjectMultiPool_h__

/** \file 
    This file contains template for generating an array of pools
    of objects of different sizes.

*/ 

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <vector>

#include "ObjectPool.h"
#include "Mutex.h"
#include "mex_assert.h"
#include "mex_tools.h"

//* This is a template for an array of ObjectPool pools (see ObjectPool.h). It
//* assumes that ObjSpecType objects can be parameterized by setting a \c size
//* parameter and that an ObjType object which was created with a given size
//* can be used for any smaller size. Accordingly, A ObjectMultiPool contains
//* an array of pools that correspond to power of 2 sizes. get() gets an
//* object with the minimal size that fits the need and put returns the object
//* to the correct pool.
//*
//* empty() returns true if no pools are defined.
//* size() returns the number of pools
//* total() returns sum of size of all pools
//* 
//* In addition to the assumptions in ObjectPool we make the following
//* assumptions:
//*   ObjTypeSpec has public member functions
//*       <tt> size_t size() const; </tt>s
//*       <tt> void setSize(size_t sz); </tt>
//*     such that after \c SetSize(sz) is called, \c size() returns \c sz.
//*   ObjType has a public member function \c size() which returns the same
//*     value as size() on the ObjSpecType object used to generate it.
//*
//* Pools are created during initialization. Each pool uses its own mutex.

template<class ObjType, class ObjSpecType>
class ObjectMultiPool
{
  // Using the alias pool_t for ObjectPool<ObjType, ObjSpecType>
  //  would make the code more readable. However, with gcc ver. 4.9.3 on
  //  Cygwin, neither of the following "using" or "typedef" commands were
  //  accepted. Hence we used the explicit templated type name.
  //
  //  using pool_t = ObjectPool<ObjType, ObjSpecType>;
  //  typedef ObjectPool<ObjType, ObjSpecType> pool_t;

public:
  ObjectMultiPool(ObjSpecType spec_, //*< Specifier for building objects
		  size_t min_size,   //*< minimum size of built objects. The
				     //*actual minimum size is the next power
				     //*of two.
		  size_t max_size_ = //*< Maximum size of built objects.
		  ObjectMultiPool<ObjType, ObjSpecType>::defaultMaxSize()
		  )
    : spec(spec_), log2_min_size(next_log2(min_size)), max_size(max_size_) 
  {
    mex_assert(min_size <= max_size_,
	       ("ObjectMultiPool:bad_args", "min_size > max_size"));
    size_t sz, next_sz = size_t(1)<<log2_min_size;
    do {
      sz = next_sz;
      spec.setSize(sz);
      pools.push_back(new ObjectPool<ObjType, ObjSpecType>(spec));
      if(sz > (max_size>>1) + (sz & 1))
	next_sz = max_size;
      else
	next_sz <<= 1;
    } while(next_sz > sz);
  }

  ~ObjectMultiPool() {
#if 0
    // This looks like the more proper way to do the loop, but for some reason
    // the compiler complains, so I used the C-like style.
    std::vector<ObjectPool<ObjType, ObjSpecType> *>::iterator p;
    for(p=pools.begin(); p != pools.end(); ++p)
      delete *p;
#else
    size_t p;
    for(p=0; p<pools.size(); p++)
      delete pools[p];
#endif
  }

  //* delete all unused elements
  void clear()
  { for(size_t p=0; p<pools.size(); p++) pools[p]->clear(); }

  size_t nOut() const {
    size_t n=0;
    for(size_t p=0; p<pools.size(); p++) 
      n += pools[p]->nOut();
    return n;
  }
  //* Verify that no objects are out and then perform clear()
  void reset() {
    mex_assert(nOut() == 0, ("ObjectMultiPool:reset", 
			  "%lu objects are out", (unsigned long)  nOut()));
    clear();
  }

      
  ObjType & get(size_t sz) {

    size_t indx = std::max(next_log2(sz),log2_min_size) - log2_min_size;

    ObjType &robj = pools[indx]->get();
 
    return robj;
  }

  void put(ObjType &robj) {
#if 0				// Other option is better
    int indx = next_log2(robj.size()) - log2_min_size;
    pools[indx]->put(robj);
#else
    ObjectPool<ObjType, ObjSpecType> *pl = 
      (ObjectPool<ObjType, ObjSpecType> *)(robj.GenericPoolObject::pool);

    pl->put(robj);
#endif
  }

  static size_t defaultMaxSize() { return ~size_t(0); }
  
  bool empty () const { return pools.empty(); }

  size_t size() const { return pools.size(); } 

  size_t total() const {
    size_t ttl = 0;
    for(size_t p=0; p<pools.size(); p++)
      if(pools[p] != NULL)
	ttl += pools[p]->size();
    return ttl;
  }

  int print(int (*fprint)(const char *fmt,...) = &printf) const {
    int n =
      fprint("ObjectMultiPool with %lu pools:", (unsigned long)pools.size());
    for(size_t p=0; p<pools.size(); p++) {
      if( pools[p] != NULL)
	n += fprint("%s0x%06lX(%4lu)",
		    (p%5 ? " ": "\n   "), (unsigned long)pools[p]->elmntSize(), 
		    (unsigned long)pools[p]->size());
      else
	n += fprint("%s    NULL      ", (p%5 ? " ": "\n   "));
    }
    n += fprint("\n");
    return n;
  }
  
private:

  ObjSpecType spec;
  int log2_min_size;
  const size_t max_size;
  std::vector<ObjectPool<ObjType, ObjSpecType> *> pools;

  int next_log2(size_t val) {
    int xpnt;
    double mnts = frexp(double(val), &xpnt);

    if(mnts == 0.5)
      xpnt--;

    // The following is to cover the case that val cannot be represented
    // exactly by a double. The multiple conditions are in order to optimize
    // out this rare case, if posssible:
    // * The first condition check (in compilation) if something like this is
    //   possible.
    // * The second condition checks if the number is large enough for this to
    //   possibly happen.
    // * The third condition (which invokes a function call) checks if it
    //   actually happened
    if(size_t(double(~size_t(0))) != ~size_t(0) && 
       xpnt > DBL_MANT_DIG &&
       size_t(ldexp(1., xpnt)) != val)
      xpnt ++;

    return xpnt;      
  }

};

#endif	/*  __ObjectMultiPool_h_- */
